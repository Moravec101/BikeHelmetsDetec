from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)  
cap.set(3, 1280)
cap.set(4, 720)


model = YOLO(r"C:\Users\marc_\Desktop\YOLO8\pt\AugNano90e.pt")

classNames = ['bicycle', 'person', 'with-helmet', 'without-helmet']

prev_frame_time = 0
new_frame_time = 0

class_colors = {
    'with-helmet': (35, 225, 50),  
    'without-helmet': (10, 50, 255),  
    'bicycle':(0, 50, 255), 
    'person': (255, 50, 255)
}


while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'with-helmet' or currentClass == 'without-helmet' and conf > 0.7:
                # Get the color for the current class from the dictionary
                myColor = class_colors.get(currentClass, (0, 0, 0))  # Default to black if class not found
                # Calculate the position for the text box below the bounding box
                text_x = max(0, x1)  
                text_y = max(y2 + 15, 35)  

                cvzone.putTextRect(img, f'{currentClass}, {conf}',
                                (text_x, text_y), scale=1.1, thickness=1, colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
