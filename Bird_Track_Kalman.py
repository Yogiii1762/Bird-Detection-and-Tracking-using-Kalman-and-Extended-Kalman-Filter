import cv2
import numpy as np
from object_detection import ObjectDetection
from kalmanfilter import KalmanFilter

od= ObjectDetection()
kf= KalmanFilter()
cap = cv2.VideoCapture("source_code\spoor.mp4")   
count =0
centre_points =[]
while True:
    ret,frame = cap.read()
    if not ret:
        break

    count += 1    
    (class_id, scores, boxes)=od.detect(frame)  

    for box in boxes:
        (x,y,w,h)= box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cx = int((x + x+ w)/2)
        cy = int((y + y+ h)/2)
        centre_points.append((cx,cy))
        cv2.circle(frame, (cx, cy), 5, (0,0,255),-1)

        for i in range(1, len(centre_points)):
            cv2.line(frame, centre_points[i-1], centre_points[i], (0,0,255), 2)

        predicted = kf.predict(cx, cy)
        cv2.circle(frame, (predicted[0], predicted[1]), 5, (255,0,0),-1)
    
    
   

    cv2.imshow("Frame",frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
    