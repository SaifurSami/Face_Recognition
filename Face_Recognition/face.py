import face_recognition

# Load the images
known_image = face_recognition.load_image_file("Project/Face_Recognition/Images/rony.jpg")
outsider_image = face_recognition.load_image_file("Project/Face_Recognition/Images/sami.jpg")

# Encode the known image
known_encodings = face_recognition.face_encodings(known_image)
if not known_encodings:
    print("No faces found in the known image.")
    exit()
rony_encoding = known_encodings[0]

# Encode the outsider image
outsider_encodings = face_recognition.face_encodings(outsider_image)
if not outsider_encodings:
    print("No faces found in the outsider image.")
    exit()
outsider_encoding = outsider_encodings[0]

# Compare faces
results = face_recognition.compare_faces([rony_encoding], outsider_encoding)
print(results)
