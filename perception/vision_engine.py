import face_recognition
python
import cv2
import numpy as np
from pathlib import Path

class VisionPerception:
    """
    Wednesday's 'Eyes'. 
    Handles face detection and identification using OpenCV and dlib.
    """
    def __init__(self, known_faces_dir: str = "./data/faces"):
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        
        self.known_encodings = []
        self.known_names = []
        self._load_known_faces()

    def _load_known_faces(self):
        """Loads and encodes all images in the known_faces directory."""
        for image_path in self.known_faces_dir.glob("*.jpg"):
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            
            if encoding:
                self.known_encodings.append(encoding[0])
                self.known_names.append(image_path.stem) # File name is the person's name

    def identify_user(self):
        """
        Captures a single frame from the webcam and attempts to identify a user.
        """
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        video_capture.release()

        if not ret:
            return "Unknown (Camera Error)"

        # Convert from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Stranger"

            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
            
            return name

        return "No face detected"

if __name__ == "__main__":
    vision = VisionPerception()
    print("Looking for faces...")
    user = vision.identify_user()
    print(f"Identified: {user}")