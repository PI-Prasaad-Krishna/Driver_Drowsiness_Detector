import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import winsound
import time

class HuggingFaceDrowsinessDetector:
    def __init__(self, model_name="chbh7051/vit-driver-drowsiness-detection", alert_threshold=0.8):
        print("[INFO] Loading Hugging Face model...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.alert_threshold = alert_threshold  # confidence threshold for alerts
        self.last_alert_time = 0  # prevent spamming alerts

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PIL
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)

            # Run inference
            inputs = self.processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = outputs.logits.softmax(dim=1)

            label_id = preds.argmax().item()
            label = self.model.config.id2label[label_id]
            conf = preds[0][label_id].item()

            # Display result
            color = (0, 255, 0) if label == "alert" else (0, 0, 255)
            cv2.putText(frame, f"{label.upper()} ({conf:.2f})", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Trigger alert if drowsy with high confidence
            if label == "drowsy" and conf >= self.alert_threshold:
                if time.time() - self.last_alert_time > 2:  # avoid rapid repeats
                    winsound.Beep(2500, 1000)  # 1s beep
                    self.last_alert_time = time.time()
                    print(f"[ALERT] Drowsiness detected at {time.ctime()} with confidence {conf:.2f}")

            cv2.imshow("HuggingFace Drowsiness Detector", frame)

            # ✅ Window closes when 'X' is clicked
            if cv2.waitKey(1) == -1:  # if no key AND window was closed
                if cv2.getWindowProperty("HuggingFace Drowsiness Detector", cv2.WND_PROP_VISIBLE) < 1:
                    break

            # Also allow quit with 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
