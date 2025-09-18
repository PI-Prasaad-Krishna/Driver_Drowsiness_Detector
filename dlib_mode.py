import cv2
import dlib
import time
import winsound  # For beep alerts (Windows)
from scipy.spatial import distance

# ---------------------------
# Function: Eye Aspect Ratio
# ---------------------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------------------------
# Parameters
# ---------------------------
EYE_AR_THRESH = 0.25              # EAR threshold for drowsiness
EYE_AR_CONSEC_FRAMES = 20         # Frames to confirm drowsiness
COUNTER = 0

# ---------------------------
# Dlib Face & Landmark Detector
# ---------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# ---------------------------
# Video Capture
# ---------------------------
cap = cv2.VideoCapture(0)

# Open log file
log_file = open("drowsiness_log.txt", "a")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw bounding boxes around eyes
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Drowsiness detection logic
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                winsound.Beep(2500, 1000)  # Beep at 2500Hz for 1s

                # Log event
                log_file.write(f"[ALERT] Drowsiness detected at {time.ctime()} | EAR: {ear:.2f}\n")
                log_file.flush()
        else:
            COUNTER = 0

    cv2.imshow("Drowsiness Detector (Dlib)", frame)

    # Quit with 'q' or ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()