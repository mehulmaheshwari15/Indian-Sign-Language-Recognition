"""
train_landmarks.py
──────────────────
Trains a lightweight classifier on MediaPipe hand-landmark coordinates
instead of raw images.  Much more accurate for visually-similar signs
(e.g. C vs F) because it uses exact finger positions rather than pixels.

Pipeline:
  data/<CLASS>/<image>.jpg
    → MediaPipe HandLandmarker → 21 landmarks (x, y, z) → 63 floats
    → normalise relative to wrist
    → Dense Neural Network (63 → 128 → 64 → num_classes)
    → saved as  isl_landmarks_model.keras
    → class list saved as  class_labels.txt
"""

import os
import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "data")
LANDMARKER  = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_SAVE  = os.path.join(BASE_DIR, "isl_landmarks_model.keras")
LABELS_FILE = os.path.join(BASE_DIR, "class_labels.txt")
EPOCHS      = 60
BATCH_SIZE  = 32

# ── Choose which classes to train on ─────────────────────────────────────────
# Set to None to auto-detect ALL folders inside data/
# Or list exactly the classes you want, e.g. ["A", "B", "C", "D"]
TRAIN_CLASSES = None   # ← EDIT THIS LINE

# ── MediaPipe Tasks setup (IMAGE mode — synchronous) ──────────────────────────
BaseOptions       = mp.tasks.BaseOptions
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=LANDMARKER),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_landmarks(image_path, detector):
    """
    Returns a flat numpy array of 63 normalised landmark values (x,y,z × 21),
    or None if no hand is detected.
    Landmarks are normalised so the wrist (index 0) is at the origin and
    the hand is scaled to a unit bounding box — making the representation
    invariant to position, scale, and (mostly) viewpoint.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    lms = result.hand_landmarks[0]          # list of 21 NormalizedLandmark
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

    # Translate so wrist is at origin
    coords -= coords[0]

    # Scale to unit bounding box
    scale = np.max(np.abs(coords)) + 1e-6
    coords /= scale

    return coords.flatten()                 # shape (63,)


# ── Extract landmarks from all images ────────────────────────────────────────
if TRAIN_CLASSES is not None:
    class_folders = sorted(TRAIN_CLASSES)
else:
    class_folders = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
print(f"Classes detected: {class_folders}\n")

X, y = [], []
label_map = {cls: i for i, cls in enumerate(class_folders)}

with HandLandmarker.create_from_options(options) as detector:
    for cls in class_folders:
        folder = os.path.join(DATA_DIR, cls)
        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        ok, skip = 0, 0
        for fname in images:
            vec = extract_landmarks(os.path.join(folder, fname), detector)
            if vec is not None:
                X.append(vec)
                y.append(label_map[cls])
                ok += 1
            else:
                skip += 1
        print(f"  {cls}: {ok} extracted, {skip} skipped (no hand detected)")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"\nDataset: {len(X)} samples, {len(class_folders)} classes")

# ── Train / Validation split ─────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
num_classes = len(class_folders)

model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=10,
                  restore_best_weights=True, verbose=1),
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# ── Save class labels ─────────────────────────────────────────────────────────
with open(LABELS_FILE, "w") as f:
    f.write("\n".join(class_folders))

print(f"\nModel saved  → {MODEL_SAVE}")
print(f"Labels saved → {LABELS_FILE}")
