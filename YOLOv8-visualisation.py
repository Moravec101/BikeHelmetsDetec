from ultralytics import YOLO
import cv2
import cvzone
import math
import os

cap = cv2.VideoCapture(r"C:\Users\marc_\Desktop\compilation.mp4")

model = YOLO(r"C:\Users\marc_\Desktop\YOLO8\pt\86epochsM.pt")

classNames = ['bicycle', 'person', 'with-helmet', 'without-helmet']

# Define colors for each class
class_colors = {
    'bicycle': (255, 165, 0),  # Orange
    'person': (255, 0, 0),     # Blue
    'with-helmet': (35, 225, 50),  # Green
    'without-helmet': (0, 0, 255)  # Red
}

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                # Get the color for the current class from the dictionary
                myColor = class_colors.get(currentClass, (0, 0, 0))  # Default to black if class not found

                # Calculate the position for the text box below the bounding box
                text_x = max(0, x1)  # Left edge of the bounding box or 0 if it's negative
                text_y = max(y2 + 15, 35)  # Place the text below the bounding box, with at least 15 pixels of space

                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                (text_x, text_y), scale=1.3, thickness=2, colorB=myColor,
                                colorT=(255, 255, 255), colorR=myColor, offset=5)


                cv2.rectangle(img, (x1, y1), (x2, y2), myColor,2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
