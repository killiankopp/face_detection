from src.face_encoder import FaceEncoder

face_encoder = FaceEncoder()
try:
    encodings = face_encoder.process_image('../artefacts/incoming/kiki2.jpg')
    for i, encoding in enumerate(encodings):
        print(f"Encodage du visage {i + 1} :", encoding)
finally:
    face_encoder.close()
