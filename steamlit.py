import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Potato Disease Classification")
st.title("🥔 Potato Disease Classification")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model3.keras")

model = load_model()

# ⚠️ CHANGE THIS IF NEEDED
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))   # ✅ Correct for Teachable Machine
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)

    predictions = model.predict(img)[0]

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}**")

    st.write("Raw prediction scores:", predictions)
