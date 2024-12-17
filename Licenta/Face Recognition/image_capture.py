import cv2
import os
from datetime import datetime
import numpy as np

PERSON_NAME = "maricel"

def create_folder(name):
    dataset_folder = "dataset"
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def initialize_kalman_filter():
    # Crearea unui obiect KalmanFilter
    kalman = cv2.KalmanFilter(4, 2)
    
    # Inițializarea stării (x, y, v_x, v_y)
    kalman.statePre = np.zeros((4, 1), np.float32)
    
    # Matricea de tranziție a stării
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    
    # Matricea de observație (locațiile x, y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    
    # Matricea de covarianță a stării
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1
    
    return kalman

def capture_photos(name):
    folder = create_folder(name)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    photo_count = 0
    kalman = initialize_kalman_filter()  # Inițializarea filtrului Kalman
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Converirea imaginii la tonuri de gri
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectarea feței folosind detectarea Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Predicția poziției feței folosind filtrul Kalman
            measurement = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
            kalman.correct(measurement)
            predicted = kalman.predict()
            
            # Desenarea locației prezise
            predicted_x, predicted_y = int(predicted[0]), int(predicted[1])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (predicted_x, predicted_y), 10, (0, 0, 255), 2)

        # Afișarea imaginii
        cv2.imshow('Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord('q'):  # Q key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)
