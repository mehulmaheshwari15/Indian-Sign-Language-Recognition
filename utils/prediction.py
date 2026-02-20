"""
Indian Sign Language Recognition - Prediction Module
=====================================================
Loads a Keras (.h5) model ONCE via a global singleton pattern and
exposes a simple predict API for OpenCV frames.

Usage (Flask integration):
    from utils.prediction import get_model, detect_and_crop_hand

    model = get_model("modelnet_model.h5")
    result = model.predict_image(frame)
    # → {"label": "A", "confidence": 0.96}
"""
from __future__ import annotations

import os

# Suppress TensorFlow GPU / info warnings before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging  # noqa: E402
from typing import Optional  # noqa: E402

import cv2  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402
import tensorflow as tf  # type: ignore  # noqa: E402

tf.get_logger().setLevel("ERROR")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class SignLanguageModel:
    """Wraps a Keras .h5 model for Indian Sign Language recognition."""

    # Placeholder labels – replace with the real class list from training
    DEFAULT_LABELS = ["A", "B", "C", "Hello", "Thankyou", "Namaste"]

    def __init__(self, model_path: str):
        """Load the Keras model from *model_path* (called ONCE at startup).

        Raises
        ------
        FileNotFoundError
            If *model_path* does not exist on disk.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}  "
                f"(cwd = {os.getcwd()})"
            )

        self.model = tf.keras.models.load_model(model_path)
        self.labels = list(self.DEFAULT_LABELS)
        print(f"✅ Model loaded: {model_path}")

    # ---- preprocessing ---------------------------------------------------
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Prepare an OpenCV frame for the model.

        Steps
        -----
        1. BGR → RGB  (OpenCV default is BGR)
        2. Resize to 224 × 224
        3. Normalize pixel values to [0, 1]
        4. Expand dims → (1, 224, 224, 3)

        Parameters
        ----------
        frame : np.ndarray
            OpenCV image of shape (H, W, 3), dtype uint8.

        Returns
        -------
        np.ndarray
            Preprocessed tensor of shape (1, 224, 224, 3), dtype float32.
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
        return img

    # ---- prediction ------------------------------------------------------
    def predict_image(self, frame: np.ndarray) -> dict:
        """Run inference on a single OpenCV frame.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV image of shape (H, W, 3).

        Returns
        -------
        dict
            ``{"label": str, "confidence": float}``
        """
        # Guard: invalid / empty frame
        if frame is None or frame.size == 0 or frame.ndim != 3:
            logger.warning("Invalid frame received – returning default prediction.")
            return {"label": "Unknown", "confidence": 0.0}

        processed = self.preprocess_image(frame)
        predictions = self.model.predict(processed, verbose=0)

        confidence = float(np.max(predictions))
        label_idx = int(np.argmax(predictions))

        # Safety: if the model has more outputs than labels
        if label_idx < len(self.labels):
            label = self.labels[label_idx]
        else:
            label = f"class_{label_idx}"

        return {"label": label, "confidence": float(f"{confidence:.4f}")}


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_model: Optional[SignLanguageModel] = None


def get_model(model_path: str = "modelnet_model.h5") -> SignLanguageModel:
    """Return (and lazily create) the global ``SignLanguageModel`` instance.

    The model is loaded **once**; subsequent calls return the cached instance.
    """
    global _model
    if _model is None:
        _model = SignLanguageModel(model_path)
    return _model


# ---------------------------------------------------------------------------
# MediaPipe hand-detection placeholder
# ---------------------------------------------------------------------------
def detect_and_crop_hand(frame: np.ndarray) -> np.ndarray:
    """Placeholder for MediaPipe hand detection.

    A teammate will replace this with actual hand-cropping logic.
    For now it returns the input frame unchanged.
    """
    return frame
