import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import urllib.request

from feature_utils import extract_features

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
LANDMARK_MODEL  = os.path.join(BASE_DIR, "isl_landmarks_model.keras")
LABELS_FILE     = os.path.join(BASE_DIR, "class_labels.txt")
DATA_DIR        = os.path.join(BASE_DIR, "data")

IMG_SIZE = 224
PADDING  = 30

# ── Auto-download hand landmarker if missing ─────────────────────────────────
if not os.path.exists(LANDMARKER_PATH):
    print("Downloading hand_landmarker.task …")
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    urllib.request.urlretrieve(url, LANDMARKER_PATH)
    print("Downloaded ✓")

# ── Load landmark-based model (preferred) or fallback to image model ──────────
if os.path.exists(LANDMARK_MODEL) and os.path.exists(LABELS_FILE):
    print(f"Loading landmark model from {LANDMARK_MODEL} …")
    model = tf.keras.models.load_model(LANDMARK_MODEL)
    with open(LABELS_FILE) as f:
        CLASS_LABELS = [line.strip() for line in f if line.strip()]
    USE_LANDMARKS = True
    print("Landmark model loaded ✓")
else:
    IMAGE_MODEL = os.path.join(BASE_DIR, "isl_model.h5")
    print(f"Landmark model not found — using image model: {IMAGE_MODEL}")
    model = tf.keras.models.load_model(IMAGE_MODEL)
    CLASS_LABELS = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    USE_LANDMARKS = False

print(f"Classes ({len(CLASS_LABELS)}): {CLASS_LABELS}")

# ── MediaPipe Tasks API ───────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

_landmarker_options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _draw_landmarks(frame, landmarks_list, frame_w, frame_h):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17),
    ]
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks_list]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 0, 128), -1)


def _bounding_box(landmarks_list, frame_w, frame_h):
    xs = [int(lm.x * frame_w) for lm in landmarks_list]
    ys = [int(lm.y * frame_h) for lm in landmarks_list]
    x_min = max(0, min(xs) - PADDING)
    y_min = max(0, min(ys) - PADDING)
    x_max = min(frame_w, max(xs) + PADDING)
    y_max = min(frame_h, max(ys) + PADDING)
    return x_min, y_min, x_max, y_max


def preprocess_image(crop):
    resized    = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def _predict_from_result(result):
    """Run ISL prediction from a HandLandmarker result. Returns (label, confidence)."""
    vec = extract_features(result)   # 195-float rich feature vector
    if vec is None:
        return None, None
    inp   = np.expand_dims(vec, axis=0)   # (1, 195)
    preds = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return CLASS_LABELS[idx], float(preds[idx]) * 100


def get_prediction(frame):
    """
    Detect hand(s) in BGR frame, run ISL model.
    Returns (annotated_frame, label_or_None, confidence_or_None).
    """
    frame_h, frame_w = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with HandLandmarker.create_from_options(_landmarker_options) as detector:
        result = detector.detect(mp_image)

    label, confidence = None, None

    if result.hand_landmarks:
        for lms in result.hand_landmarks:
            _draw_landmarks(frame, lms, frame_w, frame_h)
            x_min, y_min, x_max, y_max = _bounding_box(lms, frame_w, frame_h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

        label, confidence = _predict_from_result(result)
        if label:
            cv2.putText(frame, f"{label} ({confidence:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "No hand detected",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, label, confidence


# ── Standalone webcam loop ────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    for _ in range(5):
        cap.read()
    print("ISL Recognition started. Press 'q' to quit.")

    with HandLandmarker.create_from_options(_landmarker_options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame    = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            if result.hand_landmarks:
                for lms in result.hand_landmarks:
                    _draw_landmarks(frame, lms, frame_w, frame_h)
                    x_min, y_min, x_max, y_max = _bounding_box(lms, frame_w, frame_h)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

                lbl, conf = _predict_from_result(result)
                if lbl:
                    cv2.putText(frame, f"{lbl} ({conf:.1f}%)",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.4, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("ISL Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
