from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import os

app = FastAPI(title="Potato Disease Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load Teachable Machine model
# -----------------------
model = tf.keras.models.load_model("keras_model3.keras")

# ⚠️ MUST MATCH TEACHABLE MACHINE CLASS ORDER
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# -----------------------
# Correct preprocessing for TM
# -----------------------
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))              # ✅ REQUIRED
    image = np.array(image, dtype=np.float32)     # ❌ NO /255
    return image

@app.get("/")
def read_root():
    return {"message": "Potato classification model API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)

    predictions = model.predict(image_batch)[0]   # ✅ already softmax

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return {
        "class": predicted_class,
        "confidence": confidence,
        "raw_scores": predictions.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
