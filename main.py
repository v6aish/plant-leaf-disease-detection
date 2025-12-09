from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = r"C:\Users\vaish\Downloads\Potato\potato.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Your model's input size
IMAGE_SIZE = (256, 256)

# Class labels (edit according to your project)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return {"message": "Server is running!"}


# ------------------------------
# Prediction Function
# ------------------------------
def read_image(file) -> np.ndarray:
    """Read and preprocess image"""
    image = Image.open(BytesIO(file)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image"""
    # Read image bytes
    image_bytes = await file.read()

    # Preprocess
    img_array = read_image(image_bytes)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": round(confidence, 3)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
