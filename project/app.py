from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tf_keras.models import load_model
import time
from collections import deque, Counter
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
BASE             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH       = os.path.join(BASE, "model", "asl_model_fixed.h5")
CONFIDENCE_THRESHOLD = 0.9
FRAME_INTERVAL   = 1 / 15       # 15 FPS cap for inference
SMOOTHING_WINDOW = 10            # majority-vote over last N predictions

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
model   = load_model(MODEL_PATH, compile=False)
classes = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
print("✅ Model loaded.")

# ── Shared state ──────────────────────────────────────────────────────────────
current_prediction = {"letter": None, "confidence": 0.0, "top3": []}

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(frame):
    h, w     = frame.shape[:2]
    roi_size = min(h, w) - 20
    roi      = frame[10:10 + roi_size, 10:10 + roi_size]
    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized  = cv2.resize(gray, (28, 28)).reshape(1, 28, 28, 1) / 255.0
    return resized

def draw_overlay(frame):
    h, w     = frame.shape[:2]
    roi_size = min(h, w) - 20

    cv2.rectangle(frame, (10, 10), (10 + roi_size, 10 + roi_size), (0, 255, 180), 2)

    letter     = current_prediction["letter"]
    confidence = current_prediction["confidence"]
    top3       = current_prediction["top3"]

    if letter and confidence >= CONFIDENCE_THRESHOLD:
        cv2.putText(frame, f"{letter}  {confidence*100:.1f}%",
                    (15, 75), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 255, 180), 3)
    else:
        cv2.putText(frame, f"?  {confidence*100:.1f}%",
                    (15, 75), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 80, 255), 2)

    y = 115
    for lbl, conf in top3:
        cv2.putText(frame, f"{lbl}: {conf*100:.1f}%",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        y += 24

# ── Frame generator ───────────────────────────────────────────────────────────
def generate_frames():
    global current_prediction
    cap       = cv2.VideoCapture(0)
    last_time = 0
    history   = deque(maxlen=SMOOTHING_WINDOW)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            now = time.time()
            if now - last_time >= FRAME_INTERVAL:
                last_time = now
                try:
                    preprocessed = preprocess(frame)
                    prediction   = model.predict(preprocessed, verbose=0)[0]
                    top_idx      = prediction.argsort()[::-1][:3]
                    top3         = [(classes[i], float(prediction[i])) for i in top_idx]
                    max_conf     = float(prediction[top_idx[0]])
                    pred_class   = classes[top_idx[0]] if max_conf >= CONFIDENCE_THRESHOLD else None

                    history.append(pred_class)
                    vote     = Counter(x for x in history if x).most_common(1)
                    smoothed = vote[0][0] if vote else None

                    current_prediction = {
                        "letter":     smoothed,
                        "confidence": max_conf,
                        "top3":       top3
                    }
                except Exception as e:
                    print(f"Inference error: {e}")

            draw_overlay(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', classes=classes)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify(current_prediction)

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
