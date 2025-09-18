import cv2
import time
import winsound  # ADDED: For beep alerts

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Parameters
ALERT_TIME = 2  # Seconds eyes must be "closed" (undetected) before alert
drowsy_start = None

# Video Capture
print("Starting video stream for Haar mode...")
cap = cv2.VideoCapture(0)
window_name = "Drowsiness Detector (Haar)" # ADDED: Define window name for reuse

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        
        # If eyes are found, draw rectangles and reset timer
        if len(eyes) > 0:
            eyes_detected = True
            drowsy_start = None # Reset timer
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Logic for drowsiness detection
    if eyes_detected:
        cv2.putText(frame, "Eyes Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Eyes Not Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if drowsy_start is None:
            # Start the timer if eyes are not detected for the first time
            drowsy_start = time.time()
        elif time.time() - drowsy_start > ALERT_TIME:
            # If eyes have been undetected for long enough, sound the alert
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            winsound.Beep(2500, 1000) # ADDED: Sound alert

    cv2.imshow(window_name, frame)

    # --- MODIFIED: Improved Exit Logic ---
    key = cv2.waitKey(1) & 0xFF
    # Check for 'q' or 'ESC' to quit
    if key == ord('q') or key == 27:
        break
    # Check if the user has closed the window by clicking 'X'
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()