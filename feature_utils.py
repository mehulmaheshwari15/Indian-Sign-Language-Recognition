"""
feature_utils.py
────────────────
Shared feature extraction used by BOTH train_landmarks.py and webcam_predict.py.
MUST be kept identical in both — this file is the single source of truth.

Feature vector layout (195 floats total):
  [0:96]   Hand 1: 63 XYZ + 15 bend angles + 10 tip distances + 5 palm-to-tip + 3 orientation
  [96:192] Hand 2: same 96 features, or zeros if only 1 hand detected
  [192:195] Inter-hand wrist distance vector, or zeros if only 1 hand
"""

import numpy as np

# ── Landmark indices ──────────────────────────────────────────────────────────
FINGER_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips

# Joint-angle triplets: (parent, joint, child) → angle at joint
# Each finger chain has 4 landmarks → 3 angles
FINGER_CHAINS = [
    [0,  1,  2,  3,  4],   # Thumb
    [0,  5,  6,  7,  8],   # Index
    [0,  9, 10, 11, 12],   # Middle
    [0, 13, 14, 15, 16],   # Ring
    [0, 17, 18, 19, 20],   # Pinky
]
TRIPLETS = []
for chain in FINGER_CHAINS:
    for i in range(len(chain) - 2):
        TRIPLETS.append((chain[i], chain[i + 1], chain[i + 2]))
# 3 angles per finger × 5 fingers = 15 angles

FEAT_SIZE = 195   # total feature vector size


# ── Helpers ───────────────────────────────────────────────────────────────────
def _angle(p1, p2, p3):
    """Angle in radians at vertex p2 formed by the sequence p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def _normalize_coords(raw):
    """
    Translate so wrist (index 0) is at origin, then scale to unit bounding box.
    raw: np.float32 array (21, 3)
    Returns: (21, 3) ndarray, normalised.
    """
    c = raw - raw[0]             # translate wrist to origin
    s = np.max(np.abs(c)) + 1e-6
    return c / s


def hand_features(lms_list):
    """
    Build 96-float feature vector from one hand's 21-landmark list
    (MediaPipe NormalizedLandmark objects).

    Layout:
      0:63   – normalised XYZ coords (63)
      63:78  – finger bend angles     (15)
      78:88  – fingertip pairwise distances (10)
      88:93  – palm-to-fingertip distances   (5)
      93:96  – hand orientation vector       (3)
    """
    raw    = np.array([[lm.x, lm.y, lm.z] for lm in lms_list], dtype=np.float32)
    coords = _normalize_coords(raw)

    flat   = coords.flatten()                                  # 63

    angles = np.array(
        [_angle(coords[t[0]], coords[t[1]], coords[t[2]]) for t in TRIPLETS],
        dtype=np.float32,
    )                                                          # 15

    tips      = coords[FINGER_TIPS]
    tip_dists = np.array(
        [np.linalg.norm(tips[i] - tips[j])
         for i in range(5) for j in range(i + 1, 5)],
        dtype=np.float32,
    )                                                          # 10

    palm_dists = np.array(
        [np.linalg.norm(coords[t]) for t in FINGER_TIPS],
        dtype=np.float32,
    )                                                          # 5

    orient = coords[12] / (np.linalg.norm(coords[12]) + 1e-6)  # 3

    return np.concatenate([flat, angles, tip_dists, palm_dists, orient])  # 96


def extract_features(result):
    """
    Build full 195-float feature vector from a MediaPipe
    HandLandmarker detection result (can have 1 or 2 hands).

    Returns None if no hand is detected.
    """
    if not result.hand_landmarks:
        return None

    h1    = hand_features(result.hand_landmarks[0])           # 96

    if len(result.hand_landmarks) > 1:
        h2 = hand_features(result.hand_landmarks[1])          # 96
        c1 = np.array([[lm.x, lm.y, lm.z]
                       for lm in result.hand_landmarks[0]], dtype=np.float32)
        c2 = np.array([[lm.x, lm.y, lm.z]
                       for lm in result.hand_landmarks[1]], dtype=np.float32)
        inter = (c1[0] - c2[0]).astype(np.float32)            # 3
    else:
        h2    = np.zeros(96, dtype=np.float32)
        inter = np.zeros(3,  dtype=np.float32)

    return np.concatenate([h1, h2, inter])                    # 195
