import os

import cv2
import dlib
import mediapipe as mp
import numpy as np


def load_image(image_path) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image à l'emplacement: {image_path}")
    return image


def extract_box(detection, image) -> tuple:
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    x = int(bboxC.xmin * iw)
    y = int(bboxC.ymin * ih)
    w = int(bboxC.width * iw)
    h = int(bboxC.height * ih)
    return x, y, h, w


class FaceEncoder:
    def __init__(self, face_rec_model_path='models/dlib_face_recognition_resnet_model_v1.dat',
                 predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        face_rec_model_path = os.path.abspath(face_rec_model_path)
        predictor_path = os.path.abspath(predictor_path)

        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
        self.predictor = dlib.shape_predictor(predictor_path)
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence = 0.5)

    def process_image(self, image_path):
        image = load_image(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(rgb_image)

        face_encodings = []
        if results.detections:
            for detection in results.detections:
                x, y, h, w = extract_box(detection, image)

                rect = dlib.rectangle(x, y, x + w, y + h)

                shape = self.predictor(image, rect)

                face_encoding = np.array(self.face_rec_model.compute_face_descriptor(image, shape))
                face_encodings.append(face_encoding)

        return face_encodings

    def close(self):
        self.mp_face_detection.close()
