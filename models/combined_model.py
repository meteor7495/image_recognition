import cv2
import torch
import face_recognition
from PIL import Image
import numpy as np
import requests

class CombinedModel:
    def __init__(self):
        # Load YOLOv5 model for object detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)

    def detect_objects(self, image):
        # Convert image to RGB and PIL format for YOLOv5
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Perform object detection
        results = self.model(pil_img)
        return results

    def detect_faces(self, image):
        # Perform face detection
        face_locations = face_recognition.face_locations(image)
        return face_locations

    def predict(self, image_path):
        image = cv2.imread(image_path)
        results = {}

        # Detect faces
        face_locations = self.detect_faces(image)
        results['faces'] = [{"location": loc} for loc in face_locations]

        # Detect objects
        object_results = self.detect_objects(image)
        results['objects'] = [
            {
                "coordinates": [int(x1), int(y1), int(x2), int(y2)],
                "label": obj_name
            }
            for x1, y1, x2, y2, obj_name in zip(object_results.xyxy[0][:, 0], 
                                                object_results.xyxy[0][:, 1], 
                                                object_results.xyxy[0][:, 2], 
                                                object_results.xyxy[0][:, 3], 
                                                object_results.names)
        ]

        return results
