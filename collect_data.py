"""
collect_data.py
───────────────
Opens your webcam and automatically captures photos of a hand sign.

Usage:
    python collect_data.py C 250        → 1 hand required (default)
    python collect_data.py E 250 2      → 2 hands required (for two-hand signs)
    python collect_data.py F 250        → 1 hand required

Controls:
    SPACE  → pause / resume
    Q      → quit early
"""

import cv2
import mediapipe as mp
import os
import sys
import time
import urllib.request

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
DATA_DIR        = os.path.join(BASE_DIR, "data")
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
CAPTURE_FPS     = 4
COUNTDOWN       = 3

# ── Auto-download hand landmarker if missing ──────────────────────────────────
if not os.path.exists(LANDMARKER_PATH):
    print("Downloading hand_landmarker.task …")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        LANDMARKER_PATH,
    )
    print("Downloaded ✓")

# ── Parse args or prompt ──────────────────────────────────────────────────────
if len(sys.argv) >= 4:
    CLASS_NAME   = sys.argv[1].upper()
    TARGET_COUNT = int(sys.argv[2])
    MIN_HANDS    = int(sys.argv[3])
elif len(sys.argv) == 3:
    CLASS_NAME   = sys.argv[1].upper()
    TARGET_COUNT = int(sys.argv[2])
    MIN_HANDS    = 1
elif len(sys.argv) == 2:
    CLASS_NAME   = sys.argv[1].upper()
    TARGET_COUNT = 250
    MIN_HANDS    = 1
else:
    CLASS_NAME   = input("Enter class name (e.g. C): ").strip().upper()
    TARGET_COUNT = int(input("How many photos? [250]: ").strip() or 250)
    MIN_HANDS    = int(input("Min hands required (1 or 2)? [1]: ").strip() or 1)

# ── MediaPipe Tasks API ───────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,               # always detect up to 2
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Output folder ─────────────────────────────────────────────────────────────
OUT_DIR   = os.path.join(DATA_DIR, CLASS_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
existing  = [f for f in os.listdir(OUT_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
start_idx = len(existing)

print(f"\n📁 Saving to  : {OUT_DIR}")
print(f"📸 Existing   : {start_idx}  |  Target: {TARGET_COUNT} new photos")
print(f"👐 Min hands  : {MIN_HANDS}")
print(f"🎮 SPACE=pause  Q=quit\n")

# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)
for _ in range(10):
    cap.read()

# ── Countdown ─────────────────────────────────────────────────────────────────
t0 = time.time()
while time.time() - t0 < COUNTDOWN:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    left  = COUNTDOWN - int(time.time() - t0)
    cv2.putText(frame, f"GET READY: {left}", (80, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
    label = f"Sign: {CLASS_NAME}  ({MIN_HANDS} hand{'s' if MIN_HANDS>1 else ''})"
    cv2.putText(frame, label, (80, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    cv2.imshow("ISL Data Collector", frame)
    cv2.waitKey(1)


def draw_all_hands(frame, all_landmarks, fw, fh):
    """Draw skeleton for EVERY detected hand."""
    connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17),
    ]
    for hand_lms in all_landmarks:
        pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in hand_lms]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (255, 0, 128), -1)


# ── Main capture loop ─────────────────────────────────────────────────────────
count        = 0
paused       = False
last_cap     = time.time()
cap_interval = 1.0 / CAPTURE_FPS

with HandLandmarker.create_from_options(options) as detector:
    while count < TARGET_COUNT:
        ret, frame = cap.read()
        if not ret: break

        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        hands_found = len(result.hand_landmarks) if result.hand_landmarks else 0
        ready       = hands_found >= MIN_HANDS     # ← only capture if enough hands

        if result.hand_landmarks:
            draw_all_hands(frame, result.hand_landmarks, fw, fh)

        # ── Capture ───────────────────────────────────────────────────────────
        now      = time.time()
        captured = False
        if not paused and ready and (now - last_cap) >= cap_interval:
            fname    = os.path.join(OUT_DIR, f"{CLASS_NAME}_{start_idx+count:05d}.jpg")
            cv2.imwrite(fname, frame)
            count   += 1
            last_cap = now
            captured = True

        # ── UI ────────────────────────────────────────────────────────────────
        bar_w = int((count / TARGET_COUNT) * (fw - 40))
        cv2.rectangle(frame, (20, fh-40), (fw-20, fh-20), (40,40,40), -1)
        cv2.rectangle(frame, (20, fh-40), (20+bar_w, fh-20), (0,220,90), -1)

        if captured:
            cv2.rectangle(frame, (0,0), (fw-1,fh-1), (0,255,0), 6)

        if paused:
            status, color = "PAUSED — SPACE to resume", (0,165,255)
        elif not ready:
            status = f"Need {MIN_HANDS} hand(s) — detected {hands_found}"
            color  = (0, 80, 255)
        else:
            status = f"Capturing! [{hands_found}/{MIN_HANDS} hands detected]"
            color  = (0, 255, 0)

        cv2.putText(frame, f"Class:{CLASS_NAME}  {count}/{TARGET_COUNT}",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
        cv2.putText(frame, status,
                    (10,72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.imshow("ISL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): print("\nQuit."); break
        elif key == ord(' '):
            paused = not paused
            print("⏸ Paused" if paused else "▶ Resumed")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Captured {count} new photos for '{CLASS_NAME}'")
print(f"📁 {OUT_DIR}")
print(f"\nRetrain:  python train_landmarks.py")
