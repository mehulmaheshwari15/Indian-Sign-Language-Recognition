"""
train_landmarks.py
──────────────────
Full A-Z ISL landmark classifier trainer.

Key improvements over v1:
  • Uses 195-float rich feature vector (bend angles, tip distances, orientation, 2nd hand)
  • Helps distinguish similar signs: M/N, O/D, B/O, U/V, E/S, etc.
  • Landmark augmentation (noise + rotation + scale) → more robust
  • Class weights → balances A-G (2500+ imgs) vs H-Z (1200 imgs)
  • Deeper model with L2 regularisation + BatchNorm + Dropout
  • ReduceLROnPlateau + EarlyStopping with generous patience
"""

import os
import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from feature_utils import extract_features, FEAT_SIZE

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
LANDMARKER  = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_SAVE  = os.path.join(BASE_DIR, "isl_landmarks_model.keras")
LABELS_FILE = os.path.join(BASE_DIR, "class_labels.txt")

EPOCHS      = 120
BATCH_SIZE  = 64
L2_REG      = 5e-4

# Augmentation settings
AUG_NOISE   = 0.008   # Gaussian noise std on normalised coords
AUG_ANGLE   = 15      # max random rotation degrees

# ── MediaPipe Tasks ───────────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=LANDMARKER),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)

# ── Discover classes (normalize to uppercase) ─────────────────────────────────
raw_dirs   = sorted([d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d))])

# Map UPPERCASE_LABEL → original folder name (for os.path.join)
class_map  = {}
for folder in raw_dirs:
    lbl = folder.upper()
    if lbl not in class_map:
        class_map[lbl] = folder

class_folders = sorted(class_map.keys())
print(f"Classes ({len(class_folders)}): {class_folders}\n")

# ── Extract feature vectors ───────────────────────────────────────────────────
X, y = [], []
label_map = {cls: i for i, cls in enumerate(class_folders)}

with HandLandmarker.create_from_options(options) as detector:
    for cls in class_folders:
        folder = os.path.join(DATA_DIR, class_map[cls])
        images = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        ok, skip = 0, 0

        for fname in images:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                skip += 1
                continue
            rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)
            vec      = extract_features(result)

            if vec is not None:
                X.append(vec)
                y.append(label_map[cls])
                ok += 1
            else:
                skip += 1

        print(f"  {cls}: {ok} extracted, {skip} skipped")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"\nDataset: {len(X)} samples | {len(class_folders)} classes | {X.shape[1]} features")

# ── Train / Val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}\n")

# ── Class weights ─────────────────────────────────────────────────────────────
classes   = np.unique(y_train)
cw_vals   = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
print("Class weights computed (balanced A-G vs H-Z)\n")


# ── Augmentation helper ───────────────────────────────────────────────────────
def augment_hand_block(block63):
    """Apply noise + rotation + scale to a 63-float hand coord block."""
    c = block63.reshape(21, 3).astype(np.float32)

    c += np.random.normal(0, AUG_NOISE, c.shape).astype(np.float32)

    angle_rad      = np.random.uniform(-AUG_ANGLE, AUG_ANGLE) * np.pi / 180
    ca, sa         = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    R              = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)
    c              = (R @ c.T).T

    c             *= np.random.uniform(0.92, 1.08)

    # Re-normalise
    c -= c[0]
    s  = np.max(np.abs(c)) + 1e-6
    c /= s
    return c.flatten()


class AugSeq(tf.keras.utils.Sequence):
    """Keras Sequence: yields augmented batches (augment=True for train)."""
    def __init__(self, X, y, batch_size, augment=False):
        self.X, self.y  = X.copy(), y.copy()
        self.bs         = batch_size
        self.augment    = augment

    def __len__(self):
        return int(np.ceil(len(self.y) / self.bs))

    def __getitem__(self, idx):
        sl  = slice(idx * self.bs, (idx + 1) * self.bs)
        xb  = self.X[sl].copy()
        yb  = self.y[sl]

        if self.augment:
            for i in range(len(xb)):
                xb[i, :63]    = augment_hand_block(xb[i, :63])
                if np.any(xb[i, 96:159] != 0):       # 2nd hand present
                    xb[i, 96:159] = augment_hand_block(xb[i, 96:159])
        return xb, yb

    def on_epoch_end(self):
        idx          = np.random.permutation(len(self.y))
        self.X, self.y = self.X[idx], self.y[idx]


train_gen = AugSeq(X_train, y_train, BATCH_SIZE, augment=True)
val_gen   = AugSeq(X_val,   y_val,   BATCH_SIZE, augment=False)

# ── Model ─────────────────────────────────────────────────────────────────────
reg       = regularizers.l2(L2_REG)
n_classes = len(class_folders)

model = models.Sequential([
    layers.Input(shape=(FEAT_SIZE,)),

    layers.Dense(512, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.40),

    layers.Dense(256, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.35),

    layers.Dense(128, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.30),

    layers.Dense(64, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.20),

    layers.Dense(n_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=8,
                      min_lr=1e-6, verbose=1),
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n── Training ─────────────────────────────────────────────────────────")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
)

# ── Save labels ───────────────────────────────────────────────────────────────
with open(LABELS_FILE, "w") as f:
    f.write("\n".join(class_folders))

print(f"\n✅ Model saved  → {MODEL_SAVE}")
print(f"✅ Labels saved → {LABELS_FILE}")
