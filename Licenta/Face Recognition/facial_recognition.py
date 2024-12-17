# import face_recognition
# import cv2
# import numpy as np
# import time
# import pickle

# # Load pre-trained face encodings
# print("[INFO] Loading encodings...")
# with open("encodings.pickle", "rb") as f:
#     data = pickle.loads(f.read())
# known_face_encodings = data["encodings"]
# known_face_names = data["names"]

# # Initialize the webcam
# cap = cv2.VideoCapture(0)  # Use the default camera

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Parameters
# cv_scaler = 4  # Adjust for performance (higher = faster but less accurate)
# frame_skip = 3  # Process every nth frame to save computation
# fps_display_interval = 1  # Interval (seconds) to display FPS
# process_interval = frame_skip  # Counter for skipped frames

# # Initialize variables
# start_time = time.time()
# frame_count = 0
# fps = 0

# while True:
#     # Capture a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame from webcam.")
#         break

#     # Skip frames for faster processing
#     if process_interval % frame_skip == 0:
#         process_interval = 1  # Reset the interval counter

#         # Resize the frame to improve speed
#         resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))

#         # Convert the image to RGB color space (required by face_recognition)
#         rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

#         # Find all face locations and encodings in the frame
#         face_locations = face_recognition.face_locations(rgb_resized_frame, model='hog')
#         face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             # Check if the face matches any known faces
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"

#             # Use the known face with the smallest distance
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]
#             face_names.append(name)

#     else:
#         process_interval += 1  # Increment the frame-skip counter

#     # Draw results on the frame
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # Scale back up face locations since the frame was resized
#         top *= cv_scaler
#         right *= cv_scaler
#         bottom *= cv_scaler
#         left *= cv_scaler

#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 2)

#         # Draw a label with a name below the face
#         cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, top - 6), font, 0.7, (255, 255, 255), 1)

#     # Calculate and display FPS
#     frame_count += 1
#     elapsed_time = time.time() - start_time
#     if elapsed_time > fps_display_interval:
#         fps = frame_count / elapsed_time
#         frame_count = 0
#         start_time = time.time()

#     cv2.putText(
#         frame,
#         f"FPS: {fps:.1f}",
#         (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 0),
#         2,
#     )

#     # Show the frame
#     cv2.imshow('Video', frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) == ord("q"):
#         break

# # Clean up resources
# cap.release()
# cv2.destroyAllWindows()
import face_recognition
import cv2
import numpy as np
import time
import pickle

class KalmanFilter:
    def __init__(self):
        # Inițializăm parametrii de stare (poziție și viteză)
        self.state = np.zeros(8)  # [top, right, bottom, left, v_top, v_right, v_bottom, v_left]
        self.P = np.eye(8) * 1000  # Covarianța stării
        self.F = np.eye(8)  # Matricea de tranziție a stării
        for i in range(4):
            self.F[i, i + 4] = 1  # Adăugăm componenta de viteză
        self.Q = np.eye(8) * 0.1  # Zgomotul de proces
        self.H = np.eye(8)  # Matricea de observație
        self.R = np.eye(8) * 10  # Zgomotul de măsurare

    def predict(self):
        # Predicția stării următoare
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:4]  # Returnăm doar pozițiile (fără viteze)

    def correct(self, measurement):
        # Corectarea stării bazată pe observații
        y = measurement - self.H @ self.state  # Reziduu
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Câștigul Kalman
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.state[:4]  # Returnăm doar pozițiile corectate


# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Parameters
cv_scaler = 4  # Adjust for performance (higher = faster but less accurate)
frame_skip = 3  # Process every nth frame to save computation
fps_display_interval = 1  # Interval (seconds) to display FPS
process_interval = frame_skip  # Counter for skipped frames

# Variables for tracking
kalman_filters = []
tracked_faces = []
start_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Process every nth frame
    if process_interval % frame_skip == 0:
        process_interval = 1  # Reset interval counter

        # Detect faces and encodings
        face_locations = face_recognition.face_locations(rgb_resized_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)

        new_kalman_filters = []
        new_tracked_faces = []

        for i, face_encoding in enumerate(face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Unknown"
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Convert positions to original scale
            top, right, bottom, left = face_locations[i]
            top *= cv_scaler
            right *= cv_scaler
            bottom *= cv_scaler
            left *= cv_scaler

            # Update Kalman filter for each detected face
            kf = KalmanFilter()
            kf.correct(np.array([top, right, bottom, left, 0, 0, 0, 0]))  # Initialize
            new_kalman_filters.append(kf)
            new_tracked_faces.append((name, kf))

        # Replace old filters with new ones
        kalman_filters = new_kalman_filters
        tracked_faces = new_tracked_faces
    else:
        process_interval += 1

    # Predict positions for all tracked faces
    display_frame = frame.copy()
    for name, kf in tracked_faces:
        predicted = kf.predict()  # Predicție poziții

        top, right, bottom, left = map(int, predicted)
        # Draw rectangles and labels
        cv2.rectangle(display_frame, (left, top), (right, bottom), (244, 42, 3), 2)
        cv2.rectangle(display_frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(display_frame, name, (left + 6, top - 6), font, 0.7, (255, 255, 255), 1)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > fps_display_interval:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show the frame
    cv2.imshow('Video', display_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
