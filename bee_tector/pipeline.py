"""
This module loads the detector and subspecies models, provides utilities to
preprocess images, run predictions, and retrieve geographic context. It
combines these steps into an end-to-end pipeline.

Functions
---------
preprocess_image(img_path=...)
    Load and preprocess an image into a NumPy array.
predict_detector(x)
    Run the 3-class detector model to estimate whether an image contains
    a bumblebee.
predict_subspecies(x)
    Run the 12-class subspecies classifier to identify the subspecies.
country_context(country_code)
    Retrieve subspecies known to occur in a given country.
run_pipeline(img_path, country_code=None)
    Full pipeline: detect bumblebee, optionally classify subspecies,
    and return predictions with geographic context.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

from bee_tector.config import (
    IMAGE_SIZE,
    SUBSPECIES_DATA_DIR,
    DETECTOR_MODEL_PATH,
    SUBSPECIES_MODEL_PATH,
    DETECTOR_CLASS_NAMES,
    SUBSPECIES_CLASS_NAMES,
    IS_BUMBLEBEE_THRESHOLD,
    SUBSPECIES_HIGH_CONF_THRESHOLD
)
from bee_tector.geo import load_country_data, subspecies_seen_in

DETECTOR = load_model(DETECTOR_MODEL_PATH)
SUBSPECIES = load_model(SUBSPECIES_MODEL_PATH)
_BY_SPECIES, _NAMES = load_country_data()


# ********** PREPROCESS **********

def preprocess_image(
        img_path=os.path.join(
        SUBSPECIES_DATA_DIR, "test", "Red-tailed_Bumble_bee", "535031756.jpg"
    )):
    """
    Load and preprocess an image for prediction.

    Parameters
    ----------
    img_path : str, optional
        Path to an RGB image. Defaults to a sample file under
        SUBSPECIES_DATA_DIR for testing.

    Returns
    -------
    np.ndarray
        Array of shape (1, H, W, 3) with dtype float32.
    """
    img = load_img(img_path, target_size=IMAGE_SIZE, color_mode="rgb")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ********** DETECTOR **********

def predict_detector(x):
    """
    Run the first detector model.

    Parameters
    ----------
    x : np.ndarray
        Preprocessed image batch of shape (1, H, W, 3).

    Returns
    -------
    dict
        Dictionary with keys:
        - "probs": dict[str, float]
            Class probabilities for detector classes.
        - "label": str
            Predicted detector class label.
        - "prob": float
            Probability of the predicted detector class.
    """
    probs = DETECTOR.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "probs": {DETECTOR_CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))},
        "label": DETECTOR_CLASS_NAMES[idx],
        "prob": float(probs[idx]),
    }

# ********** SUBSPECIES **********

def predict_subspecies(x):
    """
    Run the second subspecies classifier.

    Parameters
    ----------
    x : np.ndarray
        Preprocessed image batch of shape (1, H, W, 3).

    Returns
    -------
    dict
        Dictionary with keys:
        - "class_name": str
            Predicted subspecies class name.
        - "prob": float
            Probability of the predicted subspecies.
    """
    probs = SUBSPECIES.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "class_name": SUBSPECIES_CLASS_NAMES[idx],
        "prob": float(probs[idx]),
    }


# ********** COUNTRY **********

def country_context(country_code):
    """
    Retrieve known subspecies for a given country.

    Parameters
    ----------
    country_code : str or None
        ISO 3166-1 alpha-2 country code (e.g., "GB", "US"). If None, returns [].

    Returns
    -------
    list of dict
        List of subspecies present in the country, each with
        {"class_name", "common_name", "scientific_name"}, sorted by common name.
    """
    return subspecies_seen_in(country_code, _BY_SPECIES, _NAMES)


# ********** PIPELINE **********

def run_pipeline(img_path, country_code=None):
    """
    End-to-end pipeline with optional geographic context.

    Parameters
    ----------
    img_path : str
        Path to the input image.
    country_code : str or None, optional
        ISO 2 country code for contextual species listing.

    Returns
    -------
    dict
        If detector probability for bumble_bees is below threshold:
        - {
            "stage": "no_bumblebee_detected",
            "bumblebee_prob": float,
            "threshold": float,
            "context_species_in_country": list[dict]
          }

        Otherwise (subspecies predicted):
        - {
            "stage": "subspecies_high_conf" OR "subspecies_low_conf",
            "bumblebee_prob": float,
            "prediction": {"class_name": str, "prob": float},
            "context_species_in_country": list[dict]
          }
    """
    x = preprocess_image(img_path)

    det = predict_detector(x)
    prob_bee = det["probs"].get("bumble_bees", 0.0)

    if prob_bee < IS_BUMBLEBEE_THRESHOLD:
        return {
            "stage": "no_bumblebee_detected",
            "bumblebee_prob": prob_bee,
            "threshold": IS_BUMBLEBEE_THRESHOLD,
            "context_species_in_country": country_context(country_code),
        }

    sub = predict_subspecies(x)
    high = sub["prob"] >= SUBSPECIES_HIGH_CONF_THRESHOLD

    return {
        "stage": "subspecies_high_conf" if high else "subspecies_low_conf",
        "bumblebee_prob": prob_bee,
        "prediction": sub,
        "context_species_in_country": country_context(country_code),
    }
