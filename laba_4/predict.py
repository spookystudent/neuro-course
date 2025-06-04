import cv2
import numpy as np
from tensorflow.keras.models import load_model

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

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    return image

def predict_custom_image(image_path):
    model = load_model("results/cifar10-model-v1.h5")
    image = preprocess_image(image_path)
    
    prediction = model.predict(image[np.newaxis, ...])
    predicted_class = np.argmax(prediction)
    
    print(f"\nPredicted class: {categories[predicted_class]} ({predicted_class})")
    print("All probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"{categories[i]:>12}: {prob*100:.2f}%")
    
    return image, predicted_class

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image, pred_class = predict_custom_image(sys.argv[1])
    
    plt.imshow(image)
    plt.title(f"Predicted: {categories[pred_class]}")
    plt.axis('off')
    plt.show()