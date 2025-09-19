# 🚗 Driver Drowsiness Detection

A real-time computer vision + machine learning project to detect driver drowsiness using a webcam.
The system alerts the driver when signs of sleepiness are detected.

---

## 🔹 Features

* Three detection modes:

  1. **Haar Cascades (OpenCV)** → lightweight, fast, but less robust.
  2. **Dlib Landmarks (EAR method)** → uses Eye Aspect Ratio for accurate detection.
  3. **Hugging Face Transformer (ViT model)** → deep learning-based, most robust.
* Real-time webcam feed.
* Alerts with on-screen warning + beep sound when drowsiness is detected.
* Works cross-platform (Windows/Linux/Mac).

---

## 📂 Project Structure

```
Driver_Drowsiness_Detector/
│── main.py                  # Entry point (choose mode: haar, dlib, huggingface)
│── haar_mode.py             # Haar cascade detection
│── dlib_mode.py             # Dlib EAR detection
│── huggingface_mode.py      # Hugging Face ViT model detection
│── models/
│    ├── haarcascade_eye.xml
│    └── shape_predictor_68_face_landmarks.dat
```

---

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/PI-Prasaad-Krishna/Driver_Drowsiness_Detector.git
   cd Driver_Drowsiness_Detector
   ```
   
2. Download models:

   * [Haar cascade XML](https://github.com/opencv/opencv/tree/master/data/haarcascades)
   * [Dlib landmarks model (68 points)](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
     (Place inside `models/` folder)

---

## ▶️ Usage

Run the system with one of three modes:

### Haar mode:

```bash
python main.py --mode haar
```

### Dlib mode:

```bash
python main.py --mode dlib
```

### Hugging Face mode:

```bash
python main.py --mode huggingface
```

Press **`q`** or **ESC**, or click the window **X** to exit.

---

## 📊 Comparison of Approaches

| Mode         | Technique                | Pros ✅                | Cons ❌                          |
| ------------ | ------------------------ | --------------------- | ------------------------------- |
| Haar         | Classical CV (OpenCV)    | Fast, lightweight     | Less robust, false alarms       |
| Dlib         | Eye Aspect Ratio (EAR)   | More accurate         | Slower, needs landmarks model   |
| Hugging Face | Vision Transformer (ViT) | Deep learning, robust | Heavier, needs GPU for best FPS |

---

## 📌 Future Scope

* Fine-tune Hugging Face model on custom driving datasets.
* Deploy on edge devices (e.g., Raspberry Pi, Jetson Nano).
* Integrate with IoT systems (e.g., car ignition lock, smart dashboard alert).

---
