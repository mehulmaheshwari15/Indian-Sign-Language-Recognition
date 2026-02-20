import cv2
import mediapipe as mp
import numpy as np

# Use the legacy solutions API (available in mediapipe 0.10.x via this import)
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

PADDING = 30        # pixels of padding around the bounding box
IMG_SIZE = 224      # target size expected by the model

def get_bounding_box(hand_landmarks, frame_w, frame_h, padding):
    """Calculate a padded bounding box from hand landmarks, clamped to frame bounds."""
    x_coords = [lm.x * frame_w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * frame_h for lm in hand_landmarks.landmark]

    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(frame_w, int(max(x_coords)) + padding)
    y_max = min(frame_h, int(max(y_coords)) + padding)

    return x_min, y_min, x_max, y_max

def preprocess(crop):
    """Resize to IMG_SIZE x IMG_SIZE and normalize pixels to [0.0, 1.0].
    Returns a float32 array of shape (IMG_SIZE, IMG_SIZE, 3),
    ready to be expanded to (1, IMG_SIZE, IMG_SIZE, 3) for model input.
    """
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return normalized   # shape: (224, 224, 3)

def main():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Open webcam with DirectShow backend (fixes green screen on Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Discard first few frames to allow camera to warm up
    for _ in range(10):
        cap.read()

    print("Hand detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip horizontally for a natural mirror view
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        # Convert BGR -> RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Process only the first detected hand for the crop window
            first_hand = results.multi_hand_landmarks[0]

            # Draw landmarks on all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

            # Calculate padded bounding box for first hand
            x_min, y_min, x_max, y_max = get_bounding_box(
                first_hand, frame_w, frame_h, PADDING
            )

            # Draw bounding box on main frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

            # Crop and show — guard against zero-size or invalid crops
            if x_max > x_min and y_max > y_min:
                cropped = frame[y_min:y_max, x_min:x_max]
                if cropped.size > 0:
                    cv2.imshow("Cropped Hand", cropped)

                    # --- Preprocessing pipeline (model-ready) ---
                    input_tensor = preprocess(cropped)
                    # input_tensor shape: (224, 224, 3), dtype: float32
                    # To feed into a model later:
                    #   batch = np.expand_dims(input_tensor, axis=0)  # (1, 224, 224, 3)
                    #   prediction = model.predict(batch)

            hand_count = len(results.multi_hand_landmarks)
            cv2.putText(frame, f"Hands detected: {hand_count}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Webcam - Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
