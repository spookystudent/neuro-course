import cv2
import numpy as np

def haar_face_detection(image_path):
    """Обнаружение лиц с помощью каскадов Хаара"""
    # Загрузка изображения
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Загрузка каскада Хаара
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Рисование прямоугольников вокруг лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Сохранение результата
    output_path = image_path.replace('.jpg', '_haar_detected.jpg')
    cv2.imwrite(output_path, image)
    print(f"Обнаружено {len(faces)} лиц. Результат сохранен в {output_path}")

def ssd_face_detection(image_path):
    """Обнаружение лиц с помощью SSD"""
    # Загрузка модели
    prototxt = "deploy.prototxt"
    model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    
    # Подготовка изображения для нейросети
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Обнаружение лиц
    net.setInput(blob)
    detections = net.forward()
    
    # Отрисовка результатов
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            text = f"{confidence * 100:.2f}%"
            cv2.putText(image, text, (startX, startY-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # Сохранение результата
    output_path = image_path.replace('.jpg', '_ssd_detected.jpg')
    cv2.imwrite(output_path, image)
    print(f"Обнаружено {detections.shape[2]} лиц. Результат сохранен в {output_path}")

def webcam_face_detection():
    """Обнаружение лиц с веб-камеры в реальном времени"""
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Пример использования
    image_path = "kids.jpg"
    
    print("1. Обнаружение лиц с помощью каскадов Хаара")
    haar_face_detection(image_path)
    
    print("\n2. Обнаружение лиц с помощью SSD")
    ssd_face_detection(image_path)
    
    print("\n3. Обнаружение лиц с веб-камеры (нажмите 'q' для выхода)")
    webcam_face_detection()