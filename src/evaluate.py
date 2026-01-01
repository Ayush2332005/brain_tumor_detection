import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = (224, 224)
DATA_DIR = "data/final"
MODEL_PATH = "models/brain_tumor_model.h5"

# ---------------------------
# LOAD MODEL
# (compile=False fixes custom loss loading issue)
# ---------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Model loaded successfully")

# ---------------------------
# LOAD VALIDATION DATA
# ---------------------------
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode="categorical",
    shuffle=False
)

class_names = val_ds.class_names
print("Classes:", class_names)

# ---------------------------
# PREDICTIONS
# ---------------------------
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ---------------------------
# CLASSIFICATION REPORT
# ---------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nClassification Report:")
print(report)

# ---------------------------
# SAVE RESULTS
# ---------------------------
os.makedirs("results", exist_ok=True)

# Save classification report
with open("results/classification_report.txt", "w") as f:
    f.write(report)

# Save confusion matrix plot
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

print("✅ Evaluation results saved in /results")
