from pathlib import Path

# Project root (parent of this file's folder: bee_tector/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
FULL_DATA_DIR = PROJECT_ROOT / "raw_data" / "bombus12_full"
CURATED_DATA_DIR = PROJECT_ROOT / "raw_data" / "bombus11_curated"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "inc_base.keras"
BEES_CSV_PATH = PROJECT_ROOT / "raw_data" / "bombus_inaturalist_cc.csv"
BEES_COUNTRIES_CSV = PROJECT_ROOT / "raw_data" / "bees_with_countries_raw.csv"
COUNTRIES_PREDICT = PROJECT_ROOT / "raw_data" / "bees_with_countries.csv"

# Parameters
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 16  # Adjust based on your memory capacity
SEED = 42
