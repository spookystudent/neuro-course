import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import tensorflow as tf

# Проверка и настройка GPU
gpus = tf.config.list_physical_devices('GPU')
print("GPU доступны:", tf.config.list_physical_devices('GPU'))
if gpus:
    try:
        # Разрешить TensorFlow использовать всю память GPU по мере необходимости
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        # Установить стратегию распределения
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU")
    strategy = tf.distribute.get_strategy()

# Настройки для воспроизводимости
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
np.random.seed(10)
tf.random.set_seed(10)

random_dim = 100

def load_mnist_data():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape(x_train.shape[0], 784)
    return x_train

def get_generator(optimizer):
    with strategy.scope():
        generator = Sequential([
            Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)),
            LeakyReLU(0.2),
            Dense(512),
            LeakyReLU(0.2),
            Dense(1024),
            LeakyReLU(0.2),
            Dense(784, activation='tanh')
        ])
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    with strategy.scope():
        discriminator = Sequential([
            Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)),
            LeakyReLU(0.2),
            Dropout(0.3),
            Dense(512),
            LeakyReLU(0.2),
            Dropout(0.3),
            Dense(256),
            LeakyReLU(0.2),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

def get_gan_network(discriminator, generator, optimizer):
    with strategy.scope():
        discriminator.trainable = False
        gan_input = Input(shape=(random_dim,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def plot_generated_images(epoch, generator, examples=100, dim=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=(10, 10))
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')

def train(epochs=1, batch_size=128):
    x_train = load_mnist_data()
    batch_count = x_train.shape[0] // batch_size
    if x_train.shape[0] % batch_size != 0:
        batch_count += 1 

    # Увеличьте batch_size для GPU (лучше использовать степени двойки)
    batch_size = batch_size * strategy.num_replicas_in_sync
    
    # Создаем разные оптимизаторы для каждой модели
    with strategy.scope():
        generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        gan_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        generator = get_generator(generator_optimizer)
        discriminator = get_discriminator(discriminator_optimizer)
        gan = get_gan_network(discriminator, generator, gan_optimizer)

    for e in range(1, epochs+1):
        print(f'{"-"*15} Epoch {e} {"-"*15}')
        for _ in tqdm(range(batch_count)):
            # Обучение дискриминатора
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            fake_images = generator.predict(noise, verbose=0)
            
            X = np.concatenate([real_images, fake_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9
            y_dis[batch_size:] = 0.1 
            
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            # Обучение генератора
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)


        plot_generated_images(e, generator)
        print(f'D loss: {d_loss:.4f}, G loss: {g_loss:.4f}')

if __name__ == '__main__':
    train(epochs=10, batch_size=10)