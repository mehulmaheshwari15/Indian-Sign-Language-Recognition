import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from flask import Flask, render_template, Response, jsonify

from webcam_predict import (
    model, CLASS_LABELS, preprocess_image,
    _draw_landmarks, _bounding_box,
    HandLandmarker, _landmarker_options,
    PADDING, _predict_from_result,
)

app = Flask(__name__)

# ── Settings ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 65.0   # Show label only if EMA confidence >= this
PREDICT_EVERY        = 2      # Run model every N frames (lower = faster response)
EMA_ALPHA            = 0.45   # How fast new predictions take over (0=slow, 1=instant)
NUM_CLASSES          = len(CLASS_LABELS)

# ── Shared state ──────────────────────────────────────────────────────────────
frame_lock   = threading.Lock()
latest_frame = None
latest       = {"label": None, "confidence": None}


def capture_loop():
    global latest_frame

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    for _ in range(5):   # warm-up
        cap.read()

    frame_count = 0
    last_lms_all = None

    # EMA probability vector — shape (NUM_CLASSES,)
    smooth_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    last_label = None
    last_conf  = None

    with HandLandmarker.create_from_options(_landmarker_options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame  = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            frame_count += 1

            # ── Detect hands EVERY frame ──────────────────────────────────────
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)
            hand_found = bool(result.hand_landmarks)

            if not hand_found:
                # ── Hand gone → reset EVERYTHING immediately ──────────────────
                last_lms_all = None
                smooth_probs[:] = 0.0
                last_label = None
                last_conf  = None
            else:
                last_lms_all = result.hand_landmarks

                # ── Run model every PREDICT_EVERY frames ──────────────────────
                if frame_count % PREDICT_EVERY == 0:
                    from feature_utils import extract_features
                    vec = extract_features(result)
                    if vec is not None:
                        inp      = np.expand_dims(vec, axis=0)         # (1, 195)
                        raw_probs = model.predict(inp, verbose=0)[0]

                        # EMA: blend new probs into running average
                        smooth_probs = EMA_ALPHA * raw_probs + (1 - EMA_ALPHA) * smooth_probs

                        idx  = int(np.argmax(smooth_probs))
                        conf = float(smooth_probs[idx]) * 100

                        if conf >= CONFIDENCE_THRESHOLD:
                            last_label = CLASS_LABELS[idx]
                            last_conf  = round(conf, 1)
                        else:
                            last_label = None
                            last_conf  = None

            # ── Always sync latest dict ───────────────────────────────────────
            latest["label"]      = last_label
            latest["confidence"] = last_conf

            # ── Draw landmarks and label ──────────────────────────────────────
            if last_lms_all:
                for lms in last_lms_all:
                    _draw_landmarks(frame, lms, fw, fh)
                    x1, y1, x2, y2 = _bounding_box(lms, fw, fh)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                if last_label:
                    cv2.putText(frame, f"{last_label}  {last_conf:.0f}%",
                                (14, 52), cv2.FONT_HERSHEY_SIMPLEX,
                                1.8, (0, 255, 0), 4)
                else:
                    cv2.putText(frame, "Recognising...",
                                (14, 52), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 165, 0), 2)
            else:
                cv2.putText(frame, "No hand detected",
                            (14, 52), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 80, 255), 2)

            with frame_lock:
                latest_frame = frame

    cap.release()


# ── Start background thread ───────────────────────────────────────────────────
_t = threading.Thread(target=capture_loop, daemon=True)
_t.start()


# ── MJPEG stream ──────────────────────────────────────────────────────────────
def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(1 / 30)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_LABELS)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/predict")
def predict():
    return jsonify(latest)


if __name__ == "__main__":
    print("ISL Recognition → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
