import cv2
import dlib
import time
import winsound  # For beep alerts (Windows)
from scipy.spatial import distance
import os        # NEW IMPORT
import requests  # NEW IMPORT
import bz2       # NEW IMPORT
from collections import deque
import threading # NEW IMPORT
import numpy as np # NEW IMPORT

# ---------------------------
# NEW CLASS: Webcam Video Stream
# ---------------------------
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True

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
# Function: Mouth Aspect Ratio
# ---------------------------
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])  # 51, 57
    B = distance.euclidean(mouth[2], mouth[10]) # 50, 58
    C = distance.euclidean(mouth[4], mouth[8])  # 52, 56
    D = distance.euclidean(mouth[0], mouth[6])  # 48, 54
    return (A + B + C) / (3.0 * D)

# ---------------------------
# UI Helpers
# ---------------------------
def draw_bar(img, x, y, w, h, val, max_val, color, bg_color=(50, 50, 50)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg_color, -1)
    fill_w = int(min(max(val/max_val, 0), 1) * w)
    cv2.rectangle(img, (x, y), (x+fill_w, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 200), 1)

def draw_alert(img, text, y_pos, bg_color=(0, 0, 255), text_color=(255, 255, 255)):
    (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    x_center = img.shape[1] // 2
    x_pos = x_center - (t_w // 2)
    cv2.rectangle(img, (0, y_pos - t_h - 15), (img.shape[1], y_pos + 15), bg_color, -1)
    cv2.putText(img, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)

# ---------------------------
# Parameters
# ---------------------------
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0

MOUTH_AR_THRESH = 0.70
YAWN_CONSEC_FRAMES = 10
YAWN_COUNTER = 0

PITCH_THRESH = -15
YAW_THRESH = 30
HEAD_CONSEC_FRAMES = 10
HEAD_COUNTER = 0

# History for temporal smoothing
EAR_SMOOTHING_FRAMES = 5
ear_history = deque(maxlen=EAR_SMOOTHING_FRAMES)

last_beep_time = 0

# ---------------------------
# 3D Face Model for Head Pose Estimation
# ---------------------------
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-100.0, 170.0, -135.0),     # Left eye right corner (inner)
    (100.0, 170.0, -135.0)       # Right eye left corner (inner)
], dtype="double")

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
(mStart, mEnd) = (48, 68)

# ---------------------------
# Video Capture
# ---------------------------
print("Starting video stream...")
cap = WebcamVideoStream(src=0).start()
time.sleep(1.0) # Let camera warm up

# Read one frame to get dimensions for camera matrix
_, test_frame = cap.read()
if test_frame is not None:
    size = test_frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )
    dist_coeffs = np.zeros((4,1))
else:
    camera_matrix = None
    dist_coeffs = None

# Create CLAHE object (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to improve contrast in shadows/poor lighting
    gray = clahe.apply(gray)
    
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]
        mouth = coords[mStart:mEnd]

        # ---------------------------
        # Head Pose Estimation
        # ---------------------------
        image_points = np.array([
            coords[30],     # Nose tip
            coords[8],      # Chin
            coords[36],     # Left eye left corner
            coords[45],     # Right eye right corner
            coords[39],     # Left eye right corner (inner)
            coords[42]      # Right eye left corner (inner)
        ], dtype="double")
        
        pitch, yaw, roll = 0, 0, 0
        if camera_matrix is not None:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch, yaw, roll = angles[0], angles[1], angles[2]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Temporal smoothing: add to history and calculate moving average
        ear_history.append(ear)
        smoothed_ear = sum(ear_history) / len(ear_history)
        
        mar = mouth_aspect_ratio(mouth)

        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
        for (x, y) in mouth:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        # --- HUD BACKGROUND ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # --- COLORS ---
        ear_color = (0, 0, 255) if smoothed_ear < EYE_AR_THRESH else (0, 255, 0)
        mar_color = (0, 165, 255) if mar > MOUTH_AR_THRESH else (0, 255, 0)
        pitch_color = (0, 0, 255) if pitch < PITCH_THRESH else (0, 255, 0)
        yaw_color = (0, 0, 255) if abs(yaw) > YAW_THRESH else (0, 255, 0)

        # --- TEXT & BARS ---
        # EAR
        cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        draw_bar(frame, 120, 25, 180, 15, smoothed_ear, 0.40, ear_color)
        
        # MAR
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)
        draw_bar(frame, 120, 60, 180, 15, mar, 1.0, mar_color)

        # Head Pose
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 2)

        # --- ALERTS ---
        if smoothed_ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                draw_alert(frame, "DROWSINESS ALERT!", 250, bg_color=(0, 0, 255))
                if time.time() - last_beep_time > 1.2:
                    threading.Thread(target=winsound.Beep, args=(2500, 1000), daemon=True).start()
                    last_beep_time = time.time()
        else:
            COUNTER = 0

        if mar > MOUTH_AR_THRESH:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                draw_alert(frame, "YAWNING ALERT!", 320, bg_color=(0, 165, 255))
                if time.time() - last_beep_time > 1.2:
                    threading.Thread(target=winsound.Beep, args=(2000, 1000), daemon=True).start()
                    last_beep_time = time.time()
        else:
            YAWN_COUNTER = 0

        if pitch < PITCH_THRESH or abs(yaw) > YAW_THRESH:
            HEAD_COUNTER += 1
            if HEAD_COUNTER >= HEAD_CONSEC_FRAMES:
                alert_text = "HEAD DROP ALERT!" if pitch < PITCH_THRESH else "DISTRACTED DRIVING!"
                draw_alert(frame, alert_text, 390, bg_color=(0, 0, 255))
                if time.time() - last_beep_time > 1.2:
                    threading.Thread(target=winsound.Beep, args=(3000, 1000), daemon=True).start()
                    last_beep_time = time.time()
        else:
            HEAD_COUNTER = 0

    cv2.imshow("Drowsiness Detector (Dlib)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty("Drowsiness Detector (Dlib)", cv2.WND_PROP_VISIBLE) < 1:
        break

print("Cleaning up...")
cap.stop()
cv2.destroyAllWindows()