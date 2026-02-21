import os
import sys
import shutil 

import torch
from ultralytics import YOLO

from src.vehicle.logger.logger import logger
from src.vehicle.exceptions.exception import VehicleException
from src.vehicle.entity.config_entity import ModelTrainerConfig
from src.vehicle.entity.artifacts_entity import ModelTrainerArtifact, DataIngestionArtifact
from src.vehicle.components.model_trainer.weighted_dataset import patch_yolo_with_weighted_dataset


class ModelTrainer:
    """
    Handles YOLO26 model training with:
    - Weighted dataset sampling for class imbalance
    - MuSGD optimizer (YOLO26 native)
    - Fog-aware augmentation settings
    - TF32 precision for faster GPU training
    """

    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise VehicleException(e, sys)

    def _get_data_yaml_path(self) -> str:
        """
        Locate data.yaml from the DataIngestionArtifact feature store path.
        Expected: artifacts/data_ingestion/feature_store/data.yaml
        """
        data_yaml_path = os.path.join(
            self.data_ingestion_artifact.feature_store_file_path,
            "data.yaml"
        )

        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(
                f"data.yaml not found at: {data_yaml_path}\n"
                f"Feature store path: {self.data_ingestion_artifact.feature_store_file_path}\n"
                "Please ensure DataIngestion ran successfully."
            )

        logger.info(f"Found data.yaml at: {data_yaml_path}")
        return data_yaml_path

    def _configure_torch(self):
        """Apply recommended PyTorch settings for YOLO26 training on GPU."""
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cudnn.benchmark = True
        logger.info("PyTorch TF32 precision and cuDNN benchmark enabled")

    def train(self) -> str:
        try:
            cfg = self.model_trainer_config

            self._configure_torch()

            # Inject weighted dataset before model.train() is called
            patch_yolo_with_weighted_dataset(temperature=cfg.weighted_dataset_temperature)
            logger.info(f"Injected YOLOWeightedDataset (temperature={cfg.weighted_dataset_temperature})")

            model = YOLO(cfg.model_name)
            data_yaml = self._get_data_yaml_path()

            abs_model_trainer_dir = os.path.abspath(cfg.model_trainer_dir)
            os.makedirs(abs_model_trainer_dir, exist_ok=True)
            logger.info(f"YOLO output will be saved to: {abs_model_trainer_dir}/{cfg.run_name}")

            logger.info("Starting YOLO26 model training...")
            model.train(
                data=data_yaml,
                epochs=cfg.epochs,
                imgsz=cfg.image_size,
                batch=cfg.batch_size,

                # Optimizer — MuSGD is YOLO26's native optimizer
                optimizer=cfg.optimizer,
                lr0=cfg.lr0,
                lrf=cfg.lrf,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                cos_lr=cfg.cos_lr,

                # Warmup
                warmup_epochs=cfg.warmup_epochs,
                warmup_bias_lr=cfg.warmup_bias_lr,
                warmup_momentum=cfg.warmup_momentum,

                # Augmentation
                mosaic=cfg.mosaic,
                close_mosaic=cfg.close_mosaic,
                mixup=cfg.mixup,
                copy_paste=cfg.copy_paste,
                hsv_h=cfg.hsv_h,
                hsv_s=cfg.hsv_s,
                hsv_v=cfg.hsv_v,
                scale=cfg.scale,
                translate=cfg.translate,
                fliplr=cfg.fliplr,
                degrees=cfg.degrees,
                erasing=cfg.erasing,

                # Loss weights
                # Note: dfl has no effect in YOLO26 (DFL removed from architecture)
                box=cfg.box,
                cls=cfg.cls,

                # Performance
                cache=cfg.cache,
                amp=cfg.amp,
                pretrained=cfg.pretrained,
                freeze=cfg.freeze,
                workers=cfg.workers,
                patience=cfg.patience,

                # Output
                project=abs_model_trainer_dir,
                name=cfg.run_name,
                exist_ok=cfg.exist_ok,
            )

            logger.info("YOLO26 training completed successfully")

            yolo_best_path = str(model.trainer.best)
            logger.info(f"YOLO saved best.pt at: {yolo_best_path}")

            if not os.path.exists(yolo_best_path):
                raise FileNotFoundError(
                    f"Trained model not found at: {yolo_best_path}\n"
                    "Training may have failed or been interrupted."
                )

            # Copy best.pt to canonical artifacts location
            canonical_weights_dir = os.path.join(abs_model_trainer_dir, "weights")
            os.makedirs(canonical_weights_dir, exist_ok=True)

            canonical_best_path = os.path.join(canonical_weights_dir, cfg.trained_model_file_path)
            shutil.copy2(yolo_best_path, canonical_best_path)

            logger.info(f"Copied best.pt to canonical path: {canonical_best_path}")
            return canonical_best_path
        except Exception as e:
            raise VehicleException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Initiating model trainer component")
        try:
            trained_model_path = self.train()
            model_trainer_artifact = ModelTrainerArtifact(trained_model_path=trained_model_path)

            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise VehicleException(e, sys)
