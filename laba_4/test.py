from train import load_data
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

categories = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def test_model():
    _, ds_test, _ = load_data()
    model = load_model("results/cifar10-model-v1.h5")
    
    # Оцениваем модель на тестовых данных
    test_loss, test_acc = model.evaluate(ds_test, steps=100)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    
    # Тестируем на случайном изображении
    sample = next(iter(ds_test))
    sample_image = sample[0].numpy()[0]
    sample_label = categories[sample[1].numpy()[0]]
    
    prediction = model.predict(sample_image[np.newaxis, ...])
    predicted_label = categories[np.argmax(prediction)]
    
    print(f"Predicted: {predicted_label}, Actual: {sample_label}")
    
    plt.imshow(sample_image)
    plt.title(f"Pred: {predicted_label}\nTrue: {sample_label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_model()