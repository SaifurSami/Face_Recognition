import cv2
import os
import time
import face_recognition
import numpy as np
import csv

# Constants
BASE_OUTPUT_DIR = 'saved_faces'
KNOWN_FACES_DIR = 'Images'
DESIRED_FPS = 15
PERIOD_SECONDS = 2

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a base directory to save the frames if it doesn't exist
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

# Load known faces
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(filename.split('.')[0])
    
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces(KNOWN_FACES_DIR)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

# Calculate the number of frames to save per period
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_period = int(fps * PERIOD_SECONDS)

frame_count_within_period = 0
time_folder = None

# Create a date folder and a CSV file for logging
date_folder = os.path.join(BASE_OUTPUT_DIR, time.strftime("%Y%m%d"))
if not os.path.exists(date_folder):
    os.makedirs(date_folder)
csv_file_path = os.path.join(date_folder, 'log.csv')

# Initialize CSV file with headers
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Person Name', 'Timestamp'])

index = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count_within_period += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        if time_folder is None or frame_count_within_period >= frames_per_period:
            time_folder = os.path.join(date_folder, time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(time_folder)
            frame_count_within_period = 0

        for (x, y, w, h) in faces:
            face_count = len(os.listdir(time_folder)) + 1
            face_roi = frame[y:y+h, x:x+w]
            face_filename = os.path.join(time_folder, f'face_{face_count}.jpg')
            cv2.imwrite(face_filename, face_roi)

            face_locations = [(y, x+w, y+h, x)]
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            if face_encodings:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    color = (0, 255, 0)  # Green for known
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            # Draw rectangle around the faces and display the name
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Log the details in the CSV file
            index += 1
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([index, name, timestamp])

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
