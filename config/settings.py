"""
Central configuration for the project: paths, hyperparameters, and logging.

Every other module should import paths/constants from here instead of
redefining them. Import the logger from here too:

    from config.settings import logger
"""

import os
import sys
from pathlib import Path

from loguru import logger

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# settings.py lives in config/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ASSETS_DIR = DATA_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Directories we write into must exist; assets/frontend are provided/managed
# separately so we don't create those.
for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
# NSL-KDD is the primary dataset used for training/evaluation. UNSW-NB15 is
# used as a secondary validation set (finalized in Phase 2).
PRIMARY_DATASET = "NSL-KDD"
SECONDARY_DATASET = "UNSW-NB15"

NSL_KDD_DIR = RAW_DATA_DIR / "nsl_kdd"
UNSW_NB15_DIR = RAW_DATA_DIR / "unsw_nb15"

NSL_KDD_TRAIN_PATH = NSL_KDD_DIR / "KDDTrain+.txt"
NSL_KDD_TEST_PATH = NSL_KDD_DIR / "KDDTest+.txt"
UNSW_NB15_TRAIN_PATH = UNSW_NB15_DIR / "UNSW_NB15_training-set.csv"
UNSW_NB15_TEST_PATH = UNSW_NB15_DIR / "UNSW_NB15_testing-set.csv"

# Normal-traffic label used across the datasets after preprocessing.
NORMAL_LABEL = "normal"

# --------------------------------------------------------------------------
# VAE hyperparameters
# --------------------------------------------------------------------------
LATENT_DIM = 16
HIDDEN_DIMS = [64, 32]  # encoder widths; decoder mirrors this in reverse
VAE_LEARNING_RATE = 1e-3
VAE_BATCH_SIZE = 128
VAE_EPOCHS = 50
KL_WEIGHT = 1.0  # weight of the KL term relative to reconstruction loss
SYNTHETIC_SAMPLE_COUNT = 5000

# --------------------------------------------------------------------------
# Intrusion detector settings
# --------------------------------------------------------------------------
DETECTOR_MODEL_TYPE = "random_forest"
DETECTOR_N_ESTIMATORS = 200
DETECTOR_MAX_DEPTH = None  # let trees grow fully; Random Forest's bagging controls overfitting


# NSL-KDD and UNSW-NB15 produce differently-shaped feature spaces, so every
# saved model is namespaced by dataset rather than using one fixed filename.
def vae_model_path(dataset_name: str):
    return MODELS_DIR / f"{dataset_name}_vae.pt"


def detector_model_path(dataset_name: str, augmented: bool = False):
    suffix = "_augmented" if augmented else ""
    return MODELS_DIR / f"{dataset_name}_detector{suffix}.joblib"


# --------------------------------------------------------------------------
# API settings
# --------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]

DATASET_NAMES = ("nsl_kdd", "unsw_nb15")
DATASET_DISPLAY_NAMES = {"nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15"}

# The detector variant served for live predict/simulate/traffic/alerts
# endpoints. Phase 6 found the augmented detector performs equal-or-slightly
# worse than baseline on both datasets, so baseline is what the API serves
# live by default; both remain queryable via /api/metrics for comparison.
ACTIVE_DETECTOR_VARIANT = "baseline"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.remove()  # drop loguru's default handler so we control format/level
logger.add(sys.stderr, level=LOG_LEVEL, format=LOG_FORMAT, colorize=True)
logger.add(
    LOGS_DIR / "app.log",
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    rotation="10 MB",
    retention="10 days",
)

logger.debug(f"Loaded settings. PROJECT_ROOT={PROJECT_ROOT}")
