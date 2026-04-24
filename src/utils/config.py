from pathlib import Path
import torch

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

# Dataset configuration
DATASET_NAME = "ml-100k"
DATASET_URL = f"http://files.grouplens.org/datasets/movielens/{DATASET_NAME}.zip"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds used in experiments
SEEDS = [0]

# Training defaults
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_FACTORS = 40
DEFAULT_EMBED_DIM = 32
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS_MF = 30
DEFAULT_EPOCHS_RETRIEVAL = 5
DEFAULT_EPOCHS_RANKER = 5

# Recommendation settings
DEFAULT_K = 10
DEFAULT_RETRIEVAL_TOPK = 100


def create_directories():
    """Create project directories if they do not exist."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)