from pathlib import Path

# Project root (parent of this file's folder: bee_tector/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
FULL_DATA_DIR = PROJECT_ROOT / "raw_data" / "bombus12_full"
CURATED_DATA_DIR = PROJECT_ROOT / "raw_data" / "bombus11_curated"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model"

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32  # Adjust based on your memory capacity
SEED = 42
