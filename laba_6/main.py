import os
import cv2
import mnist_loader
import network
import numpy as np

# Устанавливаем рабочую папку
# os.chdir('C:\\NeuralNetwork\\Network1')

# Загружаем обучающие данные
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Создаем сеть
net = network.Network([784, 128, 64, 10])

# Обучаем сеть (можно уменьшить количество эпох для теста)
net.SGD(training_data, epochs=1, mini_batch_size=32, eta=3.0, test_data=test_data)

# Загружаем изображение для распознавания
image = cv2.imread("digit.jpg")
digit = net.predict(image)
print("Распознанная цифра:", digit)