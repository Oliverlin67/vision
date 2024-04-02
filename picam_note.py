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
    [1726.7274948825816, 0, 1011.3633494514319],
    [0, 1726.5665555072565, 515.7561863802218],
    [0, 0, 1]], dtype=np.float32)
dist = np.array([[0.0746591, 0.3387261, -0.00489081, 0.00868447, -1.35722455]])

class_names = ['robot', 'note']
class_known_widths = [75, 35.56] # (unit: cm)
class_color = [(180, 0, 0), (0, 180, 0)] # (unit: cm)

# Initialize the capture
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

def calculate_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width * 0.01

def calculate_angle(x, y, w, h, image_width):
    target_center_x = x + w // 2
    distance_to_center = target_center_x - image_width // 2
    fov_x = 2 * math.atan(image_width / (2 * mtx[0, 0]))
    angle = distance_to_center * (fov_x / image_width)
    return angle * 100

def draw_text(
        img,
        text,
        pos=(0, 0),
        font=cv2.FONT_HERSHEY_PLAIN,
        font_scale=3,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0)
    ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x-3, y-3), (x + text_w + 6, y + text_h + 6), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

# Turn on Matplotlib interactive mode
plt.ion()

# Create Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': 'polar'})

# Lists to store detected objects
detected_objects = []

while True:
    frame = picam2.capture_array()

    results = model(frame, stream=True)
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0]*100))/100

            if confidence < 0.6:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            class_id = int(box.cls[0])

            distance = calculate_distance(class_known_widths[class_id], mtx[0, 0], w)
            angle = calculate_angle(x, y, w, h, frame.shape[1])

            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2

            draw_text(
                img=frame,
                text=class_names[class_id] + f"(conf: {confidence:.2f}%, dis: {distance:.2f}m, angle: {angle:.2f} deg)",
                pos=org,
                font=font,
                font_scale=fontScale,
                font_thickness=thickness,
                text_color=(255, 255, 255),
                text_color_bg=class_color[class_id]
            )
            # Append detected object position to list
            detected_objects.append((angle*np.pi/180, distance, class_names[class_id]))

    # Plot radar after all objects are detected
    if detected_objects:
        ax.clear()
        for angle, distance, class_name in detected_objects:
            ax.plot(angle, distance, 'ro')
            ax.text(angle, distance, class_name, fontsize=8, ha='left', va='bottom', color='black')
        fig.canvas.draw()
        detected_objects = []


    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Turn off Matplotlib interactive mode
plt.ioff()
plt.show(block=True)

cap.release()
cv2.destroyAllWindows()
