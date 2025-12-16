from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import os

app = FastAPI(title="Potato Disease Classification API")

# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load model
# -----------------------
model = tf.keras.models.load_model("potatoes.h5")

# ⚠️ MUST MATCH TRAINING ORDER
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

IMAGE_SIZE = (255, 255)  # change ONLY if training size was different

# -----------------------
# Image preprocessing (FIXED)
# -----------------------
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image, dtype=np.float32)

    # ✅ normalize ONLY ONCE
    image = image / 255.0

    return image

# -----------------------
# Routes
# -----------------------
@app.get("/")
def read_root():
    return {"message": "Potato classification model API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)

    # 🔥 Apply softmax safely
    raw_predictions = model.predict(image_batch)[0]
    predictions = tf.nn.softmax(raw_predictions).numpy()

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return {
        "class": predicted_class,
        "confidence": confidence,
        "raw_scores": predictions.tolist()  # keep for debugging
    }

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
