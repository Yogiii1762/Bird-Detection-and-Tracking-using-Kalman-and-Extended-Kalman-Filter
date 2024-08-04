import cv2
import numpy as np
from object_detection import ObjectDetection
from EKF import ExtendedKalmanFilter

# Initialize the object detection and EKF
od = ObjectDetection()
kf = ExtendedKalmanFilter(dt=1.0)

cap = cv2.VideoCapture("source_code/spoor.mp4")
count = 0
centre_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    (class_id, scores, boxes) = od.detect(frame)

    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        centre_points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Draw trajectory line
        for i in range(1, len(centre_points)):
            cv2.line(frame, centre_points[i - 1], centre_points[i], (0, 0, 255), 2)

        # Kalman filter prediction and update
        predicted = kf.predict()
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        corrected = kf.update(measurement)

        # Draw predicted point
        cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 5, (255, 0, 0), -1)
        # Draw corrected point
        cv2.circle(frame, (int(corrected[0]), int(corrected[1])), 5, (0, 255, 255), -1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
