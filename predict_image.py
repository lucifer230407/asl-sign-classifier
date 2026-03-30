"""
Test the model on a single image file.
Usage: python predict_image.py <path_to_image>
       python predict_image.py  (uses default test image)
"""
import sys
import os
import numpy as np
import cv2
from tf_keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "asl_model_fixed.h5")
classes    = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Run fix_model.py first to generate asl_model_fixed.h5")

# Image path from argument or default
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "model", "asl_test.png")

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH, compile=False)

# Preprocess
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (28, 28))
img_input   = img_resized.reshape(1, 28, 28, 1) / 255.0

# Predict
pred      = model.predict(img_input, verbose=0)[0]
top3_idx  = pred.argsort()[::-1][:3]

print("\n── Prediction Results ──────────────────")
for i, idx in enumerate(top3_idx):
    marker = "▶" if i == 0 else " "
    print(f" {marker} {classes[idx]}: {pred[idx]*100:.2f}%")
print("────────────────────────────────────────")

# Plot
plt.figure(figsize=(4, 4))
plt.imshow(img_resized, cmap='gray')
plt.title(f"Predicted: {classes[top3_idx[0]]} ({pred[top3_idx[0]]*100:.1f}%)")
plt.axis('off')
plt.tight_layout()
out = os.path.join(BASE, "model", "prediction_result.png")
plt.savefig(out)
print(f"✅ Result saved to {out}")
