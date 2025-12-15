from fastapi import FastAPI, File, UploadFile
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.responses import JSONResponse

# FastAPI app setup
app = FastAPI()

# Global variables for the model
model = None
class_names = ["Early Blight", "Late Blight", "Healthy"]
BUCKET_NAME = "your-bucket-name"  # Replace with your actual GCP bucket name
MODEL_PATH = "models/potatoes.h5"  # Path to the model in the GCP bucket

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the blob to the destination file
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model
    # Download the model from the GCP bucket
    download_blob(BUCKET_NAME, MODEL_PATH, "/tmp/potatoes.h5")
    model = tf.keras.models.load_model("/tmp/potatoes.h5")
    print("Model loaded successfully!")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle prediction requests."""
    global model
    if model is None:
        return JSONResponse(content={"error": "Model is not loaded yet"}, status_code=500)

    # Read the uploaded image
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB").resize((256, 256))

    # Convert image to numpy array and normalize it
    img_array = np.array(image) / 255.0  # Normalize the image between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return {"class": predicted_class, "confidence": confidence}
