"""
Creates a dummy Keras model (modelnet_model.h5) with:
  - Input:  (None, 224, 224, 3)
  - Output: (None, 6)   — one softmax score per label
Then runs a quick smoke-test through utils.prediction.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add project root to path so `utils` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import cv2  # type: ignore

# ── 1. Build & save a tiny dummy model ────────────────────────────────────
print("🔨 Building dummy model …")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(6, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelnet_model.h5")
model.save(MODEL_PATH)
print(f"✅ Dummy model saved → {MODEL_PATH}")

# ── 2. Test the prediction module ─────────────────────────────────────────
from utils.prediction import get_model, detect_and_crop_hand  # type: ignore  # noqa: E402

print("\n── Loading model via get_model() ──")
slm = get_model(MODEL_PATH)

# Create a fake 480×640 BGR frame (like a webcam capture)
fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print("\n── Testing detect_and_crop_hand (placeholder) ──")
cropped = detect_and_crop_hand(fake_frame)
assert cropped is fake_frame, "Placeholder should return frame unchanged"
print("   ✔ returned frame unchanged")

print("\n── Testing predict_image ──")
result = slm.predict_image(fake_frame)
print(f"   Result: {result}")
assert "label" in result and "confidence" in result
assert result["label"] in slm.labels
assert 0.0 <= result["confidence"] <= 1.0
print("   ✔ label is valid")
print("   ✔ confidence in [0, 1]")

# Edge case: invalid frame
print("\n── Testing invalid frame handling ──")
bad_result = slm.predict_image(None)
print(f"   Result (None frame): {bad_result}")
assert bad_result["label"] == "Unknown"
assert bad_result["confidence"] == 0.0
print("   ✔ gracefully handled None frame")

bad_result2 = slm.predict_image(np.array([]))
print(f"   Result (empty frame): {bad_result2}")
assert bad_result2["label"] == "Unknown"
print("   ✔ gracefully handled empty frame")

# Singleton check
print("\n── Testing singleton pattern ──")
slm2 = get_model(MODEL_PATH)
assert slm is slm2, "get_model() should return the same instance"
print("   ✔ get_model() returns same instance (singleton)")

print("\n🎉 All tests passed!")
