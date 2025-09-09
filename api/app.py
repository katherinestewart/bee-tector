from pathlib import Path
import sys

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

from bee_tector.config import FULL_DATA_DIR, CURATED_DATA_DIR, MODELS_DIR



from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the model once when the API starts (current model is the baseline)
model = load_model(MODELS_DIR / "baseline_model.keras")
class_names = ['American_Bumble_Bee',
               'Brown_belted_Bumble_Bee',
               'Buff_tailed_Bumble_Bee',
               'Common_Carder_Bumble_Bee',
               'Common_Eastern_Bumble_Bee',
               'Half_Black_Bumbnle_Bee',
               'Red_Belted_Bumble_Bee',
               'Tricolored_Bumble_Bee',
               'Two_spotted_Bumble_Bee',
               'White_tailed_Bumble_Bee',
               'Yellow_faced_Bumble_Bee'
               ]

@app.get("/")
def read_root():
    return {"message": "Bee species prediction API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))  # must match training shape
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]

    return {
        "predicted_species": pred_class,
        "confidence": float(np.max(preds))
    }
