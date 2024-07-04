import cv2
import os
import time

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a base directory to save the frames
base_output_dir = 'saved_faces'
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Set the desired frame rate (fps)
desired_fps = 30
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Frame count within period
frame_count_within_period = 0

# Calculate the number of frames to save per period (2 seconds)
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_period = int(fps * 2)

time_folder = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Increment frame count within period
    frame_count_within_period += 1

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected, create a new time folder if needed and save the face areas
    if len(faces) > 0:
        # Create a new time folder if it doesn't exist
        if time_folder is None or frame_count_within_period >= frames_per_period:
            time_folder = os.path.join(base_output_dir, time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(time_folder)
            frame_count_within_period = 0  # Reset the frame count for the period

        for (x, y, w, h) in faces:
            face_count = len(os.listdir(time_folder)) + 1
            # Extract the face from the frame
            face_roi = frame[y:y+h, x:x+w]
            # Save the face ROI in the current folder
            face_filename = os.path.join(time_folder, f'face_{face_count}.jpg')
            cv2.imwrite(face_filename, face_roi)

    # Display the resulting frame with rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
