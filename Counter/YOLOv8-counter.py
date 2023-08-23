import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture(r"C:\Users\marc_\Desktop\YOLO8\Videos\f6.mp4")

model = YOLO(r"C:\Users\marc_\Desktop\YOLO8\pt\86epochsM.pt")

classNames = ['bicycle', 'person', 'with-helmet', 'without-helmet']

mask = cv2.imread(r"C:\Users\marc_\Desktop\YOLO8\Counter\l.png")

# Tracking
tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.4)

limits = [500, 150, 500, 1000]

class_colors = {
     'with-helmet': (35, 225, 50),  # Green
    'without-helmet': (10, 50, 255)  # Red
}

totalCounth = []
totalCountw = []

idh = 0
idw = 0

while True:
    
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread(r"C:\Users\marc_\Desktop\YOLO8\Counter\300x300.png", cv2.IMREAD_UNCHANGED)


    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

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

            if conf > 0.6 and currentClass == 'with-helmet'or currentClass == 'without-helmet' :
                # Get the color for the current class from the dictionary
                myColor = class_colors.get(currentClass, (0, 0, 0))  # Default to black if class not found
                # Calculate the position for the text box below the bounding box
                text_x = max(0, x1)  
                text_y = max(y2 + 15, 35)  

                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                (text_x, text_y), scale=1.2, thickness=2, colorB=myColor,
                                colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 5)

    for result in resultsTracker:
        if currentClass == 'with-helmet' and conf > 0.1:
            x1, y1, x2, y2, idh = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
        
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h), l=6, rt=1, colorR=(0, 0, 255))
            # cvzone.putTextRect(img, f' {int(idh)}', (max(0, x1), max(35, y1)),
                            # scale=1, thickness=1, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            if limits[0] -50 < cx < limits[0] + 50 and limits[1] < cy < limits[3] :
                if totalCounth.count(idh) == 0:
                    totalCounth.append(idh)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        if currentClass == 'without-helmet' and conf > 0.1:
            x1, y1, x2, y2, idw = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
        
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h), l=6, rt=1, colorR=(255, 0, 255))
            # cvzone.putTextRect(img, f' {int(idw)}', (max(0, x1), max(35, y1)),
            #                 scale=1, thickness=1, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            if limits[0] -50 < cx < limits[0] + 50 and limits[1] < cy < limits[3] :
                if totalCountw.count(idw) == 0:
                    totalCountw.append(idw)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(img,str(len(totalCounth)),(190,105),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.putText(img,str(len(totalCountw)),(190,245),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)


    cv2.imshow('Image', img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)