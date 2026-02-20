import os 

# ─────────────────────────────────────────────
# Data Ingestion Constants
# ─────────────────────────────────────────────
ARTIFACTS_DIR: str = "artifacts"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE: str = "feature_store"
DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1pVYOFZAG3BWFc_Vey-1vpzoyeRs0cahE/view?usp=sharing"

# ─────────────────────────────────────────────
# Data Validation Constants
# ─────────────────────────────────────────────
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_STATUS_FILE = 'status.txt'
DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "valid", "test", "data.yaml"]

# ─────────────────────────────────────────────
# Data Preprocessing Constants
# ─────────────────────────────────────────────
DATA_PREPROCESSING_DIR_NAME: str = "data_preprocessing"
DATA_PREPROCESSING_TRAIN_IMAGES: str = os.path.join("train", "images")
DATA_PREPROCESSING_TRAIN_LABELS: str = os.path.join("train", "labels")
OVERSAMPLING_MINORITY_PERCENTILE: float = 40.0
OVERSAMPLING_DUPLICATE_COUNT: int = 2

# ─────────────────────────────────────────────
# Model Trainer Constants
# ─────────────────────────────────────────────
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "best.pt"
MODEL_DEFAULT_NAME: str = "yolo26m.pt"
MODEL_DEFAULT_EPOCHS: int = 150
MODEL_DEFAULT_BATCH_SIZE: int = 64
MODEL_DEFAULT_IMAGE_SIZE: int = 640

# ─────────────────────────────────────────────
# Model Evaluation Constants
# ─────────────────────────────────────────────
EVALUATION_CONF_DEFAULT: float = 0.001
EVALUATION_IOU_DEFAULT: float = 0.6
EVALUATION_IOU_SWEEP: list = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
EVALUATION_CONF_SWEEP: list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
