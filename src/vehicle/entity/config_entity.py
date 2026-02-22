import os 
from datetime import datetime 
from dataclasses import dataclass
from src.vehicle.constants.training_pipeline.constant import * 
from src.vehicle.constants.training_pipeline.model_trainer_constants import *
from src.vehicle.constants.training_pipeline.model_eval_constants import *

@dataclass 
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACTS_DIR

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig() 



@dataclass 
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifacts_dir,
                                           DATA_INGESTION_DIR_NAME)
    
    feature_store_file_path: str = os.path.join(data_ingestion_dir,
                                                DATA_INGESTION_FEATURE_STORE)
    
    data_download_url: str = DATA_DOWNLOAD_URL


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifacts_dir,
                                            DATA_VALIDATION_DIR_NAME)
    
    valid_status_file_dir: str = os.path.join(data_validation_dir,
                                              DATA_VALIDATION_STATUS_FILE)
    
    required_file_list = DATA_VALIDATION_ALL_REQUIRED_FILES

@dataclass
class DataPreprocessingConfig:
    data_preprocessing_dir: str = os.path.join(training_pipeline_config.artifacts_dir,
                                          DATA_PREPROCESSING_DIR_NAME)
    
    preprocessed_train_images_dir: str = os.path.join(data_preprocessing_dir,
                                                      DATA_PREPROCESSING_TRAIN_IMAGES)
    
    preprocessed_train_labels_dir: str = os.path.join(data_preprocessing_dir,
                                                      DATA_PREPROCESSING_TRAIN_LABELS)
    minority_percentile: float = OVERSAMPLING_MINORITY_PERCENTILE
    
    duplicate_count: int = OVERSAMPLING_DUPLICATE_COUNT
    n_augments: int =  2 


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        MODEL_TRAINER_DIR_NAME
    )
    trained_model_file_path: str = MODEL_TRAINER_TRAINED_MODEL_NAME
    epochs: int = MODEL_DEFAULT_EPOCHS
    batch_size: int = MODEL_DEFAULT_BATCH_SIZE
    image_size: int = MODEL_DEFAULT_IMAGE_SIZE
    model_name: str = MODEL_DEFAULT_NAME

    # Optimizer
    optimizer: str = MODEL_TRAINER_OPTIMIZER
    lr0: float = MODEL_TRAINER_LR0
    lrf: float = MODEL_TRAINER_LRF
    momentum: float = MODEL_TRAINER_MOMENTUM
    weight_decay: float = MODEL_TRAINER_WEIGHT_DECAY
    cos_lr: bool = MODEL_TRAINER_COS_LR

    # Warmup
    warmup_epochs: float = MODEL_TRAINER_WARMUP_EPOCHS
    warmup_bias_lr: float = MODEL_TRAINER_WARMUP_BIAS_LR
    warmup_momentum: float = MODEL_TRAINER_WARMUP_MOMENTUM

    # Augmentation
    mosaic: float = MODEL_TRAINER_MOSAIC
    close_mosaic: int = MODEL_TRAINER_CLOSE_MOSAIC
    mixup: float = MODEL_TRAINER_MIXUP
    copy_paste: float = MODEL_TRAINER_COPY_PASTE
    hsv_h: float = MODEL_TRAINER_HSV_H
    hsv_s: float = MODEL_TRAINER_HSV_S
    hsv_v: float = MODEL_TRAINER_HSV_V
    scale: float = MODEL_TRAINER_SCALE
    translate: float = MODEL_TRAINER_TRANSLATE
    fliplr: float = MODEL_TRAINER_FLIPLR
    degrees: float = MODEL_TRAINER_DEGREES
    erasing: float = MODEL_TRAINER_ERASING

    # Loss weights
    box: float = MODEL_TRAINER_BOX_LOSS
    cls: float = MODEL_TRAINER_CLS_LOSS

    # Performance
    cache: str = MODEL_TRAINER_CACHE
    amp: bool = MODEL_TRAINER_AMP
    compile: bool = MODEL_TRAINER_COMPILE
    pretrained: bool = MODEL_TRAINER_PRETRAINED
    freeze: int = MODEL_TRAINER_FREEZE
    workers: int = MODEL_TRAINER_WORKERS
    patience: int = MODEL_TRAINER_PATIENCE

    # Weighted dataset
    weighted_dataset_temperature: float = WEIGHTED_DATASET_TEMPERATURE

    # Output
    run_name: str = MODEL_TRAINER_RUN_NAME
    exist_ok: bool = MODEL_TRAINER_EXIST_OK


@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir,
        EVALUATION_DIR_NAME
    )
    evaluation_report_path: str = os.path.join(model_evaluation_dir, EVALUATION_REPORT_FILE)
    plots_dir: str = os.path.join(model_evaluation_dir, EVALUATION_PLOTS_DIR)

    conf_default: float = EVALUATION_CONF_DEFAULT
    iou_default: float = EVALUATION_IOU_DEFAULT
    iou_sweep: list = None          # set in __post_init__
    conf_sweep: list = None

    split: str = EVALUATION_SPLIT

    def __post_init__(self):
        if self.iou_sweep is None:
            self.iou_sweep = EVALUATION_IOU_SWEEP
        if self.conf_sweep is None:
            self.conf_sweep = EVALUATION_CONF_SWEEP