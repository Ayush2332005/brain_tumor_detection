import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = (224, 224)
DATA_DIR = "data/final/val"
MODEL_PATH = "models/brain_tumor_model.h5"
OUTPUT_CSV = "results/val_predictions.csv"

# ---------------------------
# LOAD MODEL (IMPORTANT FIX)
# ---------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False   # ✅ FIX for custom focal loss
)

print("✅ Model loaded successfully")

# ---------------------------
# LOAD CLASS NAMES
# ---------------------------
class_names = sorted(os.listdir(DATA_DIR))
print("Classes:", class_names)

# ---------------------------
# PREDICT ON VALIDATION SET
# ---------------------------
results = []

for actual_class in class_names:
    class_path = os.path.join(DATA_DIR, actual_class)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Load and preprocess image
        img = load_img(img_path, target_size=IMG_SIZE)
        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # Predict
        preds = model.predict(img_arr, verbose=0)[0]
        pred_idx = np.argmax(preds)

        results.append({
            "image": img_name,
            "actual": actual_class,
            "predicted": class_names[pred_idx],
            "confidence": round(float(preds[pred_idx]), 4)
        })

# ---------------------------
# SAVE RESULTS
# ---------------------------
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Predictions saved to {OUTPUT_CSV}")
print(df.head())
