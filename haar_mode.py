import cv2
import time

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

blink_counter = 0
drowsy_start = None
ALERT_TIME = 2  # seconds eyes must be closed before alert

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 1:  # At least one eye detected
            eyes_detected = True
            drowsy_start = None
            cv2.putText(frame, "Eyes Open", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if not eyes_detected:
        if drowsy_start is None:
            drowsy_start = time.time()
        elif time.time() - drowsy_start > ALERT_TIME:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    cv2.imshow("Drowsiness Detector (Haar)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
