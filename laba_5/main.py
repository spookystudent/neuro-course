# Создание LSTM-нейросети для генерации текста с использованием библиотеки PyTorch.

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Функции подготовки данных для сети
TRAIN_TEXT_FILE_PATH = 'train_text.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    print(sorted_chars)
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])

    return sequence, char_to_idx, idx_to_char


from train_text import text
sequence, char_to_idx, idx_to_char = text_to_seq(text*2)

# Генерация пакетов (Batch) из текста
SEQ_LEN = 10
BATCH_SIZE = 2048

def get_batch(sequence):
    trains = []
    targets = []
    for _ in range(BATCH_SIZE):
        batch_start = np.random.randint(0, len(sequence) - SEQ_LEN)
        chunk = sequence[batch_start:batch_start + SEQ_LEN]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)

# Функция, генерирующая текст
def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=256, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    hidden = (hidden[0].to(device), hidden[1].to(device))

    _, hidden = model(train, hidden)

    input = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(input.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        input = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text

# Создаем класс нейросети
class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (h_n, c_n) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (h_n, c_n)

    def init_hidden(self, batch_size=1):
        # Инициализация на том же устройстве, что и модель
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        )

# Создаем нейросеть и обучаем ее
device = torch.device('cuda')
model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5,
    verbose=True,
    factor=0.5
)

n_epochs = 500
loss_avg = []

for epoch in range(n_epochs):
    model.train()
    hidden = model.init_hidden(BATCH_SIZE)  # Теперь hidden будет на правильном устройстве
    
    train, target = get_batch(sequence)
    train = train.permute(1, 0, 2).to(device)
    target = target.permute(1, 0, 2).to(device)
    
    output, hidden = model(train, hidden)
    loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_avg.append(loss.item())
    if len(loss_avg) >= 50:
        mean_loss = np.mean(loss_avg)
        print(f'Loss: {mean_loss}')
        scheduler.step(mean_loss)
        loss_avg = []
        model.eval()
        predicted_text = evaluate(model, char_to_idx, idx_to_char)
        print(predicted_text)
print('\n\n\n\n')
# Генерируем текст
model.eval()
print(evaluate(
    model,
    char_to_idx,
    idx_to_char,
    temp=0.45,
    prediction_len=1000,
    start_text="Вот и кончилась жизнь в этом доме, "
))