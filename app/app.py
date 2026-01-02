import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "normal",
    "pituitary_tumor"
]

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "models/brain_tumor_model.h5",
        compile=False
    )
    return model

model = load_model()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to classify the tumor type.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# PREDICTION
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.markdown("---")
    st.subheader("🧪 Prediction Result")
    st.write(f"**Tumor Type:** `{predicted_class}`")
    st.write(f"**Confidence:** `{confidence:.2%}`")
