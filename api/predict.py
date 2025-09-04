import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

from bee_tector.config import (
    BEST_MODEL_PATH,
    FULL_DATA_DIR,
    IMAGE_SIZE,
    )


def preprocess_image(
        img_path=os.path.join(
        FULL_DATA_DIR, "test", "Red-tailed_Bumble_bee", "535031756.jpg"
    )):
    """
    Load and preprocess image for prediction.
    """
    img = load_img(img_path, target_size=IMAGE_SIZE, color_mode="rgb")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_best_model():
    """
    Load best model for prediction.
    """
    model = load_model(BEST_MODEL_PATH)
    return model


def predict(best_model, img_array):
    """
    Make prediction with best model from preprocessed user image.
    """
    id_to_class = {
        0: 'American_Bumble_Bee',
        1: 'Brown-belted_Bumble_Bee',
        2: 'Buff-tailed_Bumble_Bee',
        3: 'Common_Carder_Bumble_Bee',
        4: 'Common_Eastern_Bumble_Bee',
        5: 'Half-black_Bumble_Bee',
        6: 'Red-belted_Bumble_Bee',
        7: 'Red-tailed_Bumble_Bee',
        8: 'Tricolored_Bumble_Bee',
        9: 'Two-spotted_Bumble_Bee',
        10: 'White-tailed_Bumble_Bee',
        11: 'Yellow-faced_Bumble_Bee'
    }

    preds = best_model.predict(img_array)
    pred_index = np.argmax(preds, axis=1)[0]
    pred_class = id_to_class[pred_index]
    confidence = round(float(preds[0][pred_index]) * 100, 2)

    return {"class": pred_class, "confidence": f"{confidence}%"}
