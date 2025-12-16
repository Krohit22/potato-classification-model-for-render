from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()


Model = tf.keras.models.load_model("./potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get('/')
def read_root():
    return {'message': 'potatos classification model API'}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,0)
    predictions = Model.predict(image_batch)
    prediction_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(prediction_class, confidence)
    return{
        "class" : prediction_class,
        "confidence" : float(confidence)
    }

    