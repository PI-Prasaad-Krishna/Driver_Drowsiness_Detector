import cv2
import dlib
import time
import winsound  # For beep alerts (Windows)
from scipy.spatial import distance
import os        # NEW IMPORT
import requests  # NEW IMPORT
import bz2       # NEW IMPORT

# ---------------------------
# NEW FUNCTION: Download dlib model
# ---------------------------
def download_dlib_model():
    """
    Checks for the dlib model, downloads and extracts it if it doesn't exist.
    """
    model_dir = "models"
    model_path = os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")

    # Check if the model file already exists
    if os.path.exists(model_path):
        print("[INFO] Dlib model already exists.")
        return

    # Create the 'models' directory if it doesn't exist
    print("[INFO] 'models' directory not found. Creating it...")
    os.makedirs(model_dir, exist_ok=True)

    # Download the compressed model
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    print(f"[INFO] Downloading dlib model from {url}...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Decompress and save the file
        print(" decompressing and saving model...")
        with open(model_path, "wb") as f_out:
            decompressor = bz2.BZ2Decompressor()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f_out.write(decompressor.decompress(chunk))
        print(f"✅ Model saved to {model_path}")
    else:
        print(f"[INFO] Failed to download model. Status code: {response.status_code}")
        exit()

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
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0

# ---------------------------
# NEW: Call the download function before loading the model
# ---------------------------
download_dlib_model()

# ---------------------------
# Dlib Face & Landmark Detector
# ---------------------------
print(" dlib face and landmark detectors...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# ---------------------------
# Video Capture
# ---------------------------
print("Starting video stream...")
cap = cv2.VideoCapture(0)

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

        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                winsound.Beep(2500, 1000)
        else:
            COUNTER = 0

    cv2.imshow("Drowsiness Detector (Dlib)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty("Drowsiness Detector (Dlib)", cv2.WND_PROP_VISIBLE) < 1:
        break

print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()