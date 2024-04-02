import cv2
import numpy as np
import math
from ultralytics import YOLO
import matplotlib.pyplot as plt
from picamera2 import Picamera2

# Load a YOLO model
model = YOLO('YOLOv8nNORO.pt')

# Camera parameters
mtx = np.array([
    [1081.1243429774236, 0, 585.43772498095],
    [0, 1074.2970305057204, 382.31948322396977],
    [0, 0, 1]], dtype=np.float32)
dist = np.array([[1.07342905e-01, -7.92326588e-01, 2.72175242e-04, -5.00060782e-03, 1.03237106e+00]])


class_names = ['robot', 'note']
class_known_widths = [75, 35.56] # (unit: cm)
class_color = [(180, 0, 0), (0, 180, 0)] # (unit: cm)

# Initialize the capture
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (640, 270)}))
picam2.start()

def calculate_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width * 0.01

def calculate_angle(x, y, w, h, image_width):
    target_center_x = x + w // 2
    distance_to_center = target_center_x - image_width // 2
    fov_x = 2 * math.atan(image_width / (2 * mtx[0, 0]))
    angle = distance_to_center * (fov_x / image_width)
    return angle * 100

# Lists to store detected objects
detected_objects = []

while True:
    frame = picam2.capture_array()

    results = model(frame, stream=True)
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0]*100))/100

            """
            if confidence < 0.6:
                continue
            """

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            class_id = int(box.cls[0])

            distance = calculate_distance(class_known_widths[class_id], mtx[0, 0], w)
            angle = calculate_angle(x, y, w, h, frame.shape[1])

            print(class_names[class_id] + f"(conf: {confidence:.2f}%, dis: {distance:.2f}m, angle: {angle:.2f} deg)")
            # Append detected object position to list
            #detected_objects.append((angle*np.pi/180, distance, class_names[class_id]))

    #cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break