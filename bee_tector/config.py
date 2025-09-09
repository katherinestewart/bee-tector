"""
This module defines project paths, parameters, thresholds, and class names
used across the BeeTector application. It centralizes constants for both
training and prediction.
"""

from pathlib import Path

# Project root (parent of this file's folder: bee_tector/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
SUBSPECIES_DATA_DIR = PROJECT_ROOT / "raw_data" / "bombus12_full"
DETECTOR_DATA_DIR = PROJECT_ROOT / "raw_data" / "bumble_ternary"
MODELS_DIR = PROJECT_ROOT / "models"
SUBSPECIES_MODEL_PATH = PROJECT_ROOT / "models" / "inc_base.keras"
DETECTOR_MODEL_PATH = PROJECT_ROOT / "models" / "first_layer_base.keras"
BEES_CSV_PATH = PROJECT_ROOT / "raw_data" / "bombus_inaturalist_cc.csv"
BEES_COUNTRIES_CSV = PROJECT_ROOT / "raw_data" / "bees_with_countries_raw.csv"
SUBSPECIES_COUNTRIES_CSV = PROJECT_ROOT / "raw_data" / "bees_with_countries.csv"

# Parameters
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 16
SEED = 42

# Thresholds
IS_BUMBLEBEE_THRESHOLD = 0.75
SUBSPECIES_HIGH_CONF_THRESHOLD = 0.75

# Class names
DETECTOR_CLASS_NAMES   = ["bumble_bees", "lookalikes", "others"]
SUBSPECIES_CLASS_NAMES = [
    "American_Bumble_Bee",
    "Brown-belted_Bumble_Bee",
    "Buff-tailed_Bumble_Bee",
    "Common_Carder_Bumble_Bee",
    "Common_Eastern_Bumble_Bee",
    "Half-black_Bumble_Bee",
    "Red-belted_Bumble_Bee",
    "Red-tailed_Bumble_Bee",
    "Tricolored_Bumble_Bee",
    "Two-spotted_Bumble_Bee",
    "White-tailed_Bumble_Bee",
    "Yellow-faced_Bumble_Bee",
]
