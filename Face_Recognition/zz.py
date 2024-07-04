import os
import face_recognition

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(filename)

    return known_encodings, known_names

def compare_faces_in_folder(faces_folder, known_faces_dir):
    known_encodings, known_names = load_known_faces(known_faces_dir)

    results = []

    i = 0
    for subdir, dirs, files in os.walk(faces_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                print(f"images: {i}\r", end="")
                i = i+1
                face_path = os.path.join(subdir, filename)
                face_image = face_recognition.load_image_file(face_path)
                face_encodings = face_recognition.face_encodings(face_image)

                if face_encodings:
                    face_encoding = face_encodings[0]

                    match_found = False
                    for known_encoding, known_name in zip(known_encodings, known_names):
                        match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
                        if match:
                            results.append((face_path, known_name, "Match"))
                            match_found = True
                            break
                    
                    if not match_found:
                        results.append((face_path, "Unknown", "No Match"))

    return results

# Example usage
faces_folder = 'saved_faces'
known_faces_dir = 'Images'
comparison_results = compare_faces_in_folder(faces_folder, known_faces_dir)

# Print results with proper spacing
print(f"{'Face Image':<45} {'Known Face':<10} {'Verdict':<5}")
print("="*90)
for face_path, known_name, verdict in comparison_results:
    print(f"{face_path:<45} {known_name:<20} {verdict:<5}")
