# 🤟 Indian Sign Language Recognition

Real-time Indian Sign Language (ISL) alphabet recognition using **MediaPipe hand landmarks** and a **deep neural network**. Recognises all **26 letters (A–Z)** from a live webcam feed with **98.27% accuracy**.

![ISL Recognition Demo](https://img.shields.io/badge/Accuracy-98.27%25-brightgreen) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

---

## ✨ Features

- 🎥 **Real-time webcam** sign detection via browser
- 🖐️ **MediaPipe hand tracking** with skeleton overlay
- 🧠 **195-feature rich extraction** (bend angles, fingertip distances, palm orientation, 2-hand support)
- ⚡ **EMA smoothing** for stable, flicker-free predictions
- 🎨 **Premium dark UI** with live confidence bar
- 📊 **26 classes** (A–Z) with class-weighted training

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Indian-Sign-Language-Recognition.git
cd Indian-Sign-Language-Recognition
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the App

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

> The trained model (`isl_landmarks_model.keras`) is included — **no training needed** to run the app!

---

## 📂 Project Structure

```
├── app.py                      # Flask server + webcam capture loop
├── webcam_predict.py           # Model loading + inference
├── feature_utils.py            # 195-float feature extraction (shared)
├── train_landmarks.py          # Training script (only if retraining)
├── collect_data.py             # Webcam data collection tool
├── isl_landmarks_model.keras   # Trained model (98.27% accuracy)
├── class_labels.txt            # A–Z class labels
├── hand_landmarker.task        # MediaPipe hand detector model
├── requirements.txt            # Python dependencies
└── templates/
    └── index.html              # Web UI
```

---

## 📦 Dataset

The training dataset (~0.57 GB, ~42,000 images) is too large for GitHub.

**📥 Download from Google Drive:**
👉 [**ISL Dataset (Google Drive)**](https://drive.google.com/drive/folders/1GYyVNiVdhzvV48ZbQKHEaUNNt9iK0yT_?dmr=1)

After downloading, extract into the project root:
```
Indian-Sign-Language-Recognition/
├── data/
│   ├── A/   (2426 images)
│   ├── B/   (2528 images)
│   ├── ...
│   └── Z/   (1200 images)
```

> **Note:** You only need the dataset if you want to **retrain** the model. The pre-trained model is already included in the repo.

---

## 🔄 Retrain (Optional)

If you want to retrain with new/additional data:

```bash
# Collect new data (e.g., 300 photos of letter R with 2 hands)
python collect_data.py R 300 2

# Retrain the model
python train_landmarks.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Hand Detection | MediaPipe Hand Landmarker |
| ML Model | TensorFlow/Keras Dense NN |
| Feature Vector | 195 floats (angles + distances + orientation) |
| Backend | Flask |
| Frontend | HTML/CSS/JS |
| Webcam | OpenCV |

---

## 📊 Model Details

- **Architecture:** Dense 512→256→128→64→26 with BatchNorm + Dropout + L2
- **Features:** 195-float vector per sample (vs raw 63 XYZ coords)
- **Training:** Class-weighted, landmark augmentation (noise + rotation + scale)
- **Validation Accuracy:** 98.27%
- **Anti-overfitting:** EarlyStopping (patience=20), ReduceLROnPlateau

---

## 👥 Team

Built for Hackathon 2026 🚀