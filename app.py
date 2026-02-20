"""
SignSight AI — Flask Application
Serves the frontend and handles /predict_image API endpoint.
"""

from flask import Flask, render_template, request, jsonify
import cv2  # type: ignore
import numpy as np  # type: ignore

from utils.prediction import get_model, detect_and_crop_hand

app = Flask(__name__)

# Load model once at startup
model = get_model("modelnet_model.h5")


@app.route("/")
def index():
    """Serve the main SignSight AI page."""
    return render_template("index.html")


@app.route("/predict_image", methods=["POST"])
def predict_image():
    """Accept an uploaded image and return sign language prediction.

    Expects: multipart/form-data with field 'image'
    Returns: {"label": str, "confidence": float}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    file_bytes = file.read()

    if not file_bytes:
        return jsonify({"error": "Empty image file"}), 400

    # Decode image from bytes
    frame = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Hand detection placeholder (teammate will replace)
    frame = detect_and_crop_hand(frame)

    # Run prediction
    result = model.predict_image(frame)
    return jsonify(result)


if __name__ == "__main__":
    print("🚀 Starting SignSight AI on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
