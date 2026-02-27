import os 
import sys 
import shutil 

import cv2 
import yaml 
import numpy as np 
import albumentations as A 

from src.vehicle.logger.logger import logger 
from src.vehicle.exceptions.exception import VehicleException
from src.vehicle.entity.config_entity import DataPreprocessingConfig
from src.vehicle.entity.artifacts_entity import (DataIngestionArtifact,
                                                 DataPreprocessingArtifact)

class DataPreprocessing:
    """
    Handles two preprocessing steps before training:
    1. Minority class oversampling  — duplicates images containing rare classes
    2. Albumentations augmentation  — generates fog-aware augmented copies

    Output is stored under artifacts/data_preprocessing/train/images and labels.
    The original feature_store data is never modified.
    data.yaml is updated to point to the new preprocessed train directory.
    """
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_preprocessing_config: DataPreprocessingConfig = DataPreprocessingConfig()):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_preprocessing_config = data_preprocessing_config

            self.feature_store = data_ingestion_artifact.feature_store_file_path

            # Source dirs — original data from feature store, never modified
            self.src_images_dir = os.path.join(self.feature_store, "train", "images")
            self.src_labels_dir = os.path.join(self.feature_store, "train", "labels")

            # Output dirs — under artifacts/data_preprocessing
            self.out_images_dir = data_preprocessing_config.preprocessed_train_images_dir
            self.out_labels_dir = data_preprocessing_config.preprocessed_train_labels_dir

            self.data_yaml_path = os.path.join(self.feature_store, "data.yaml")

        except Exception as e:
            raise VehicleException(e, sys)
        
    def _copy_train_split(self):
        """
        Copy original train images and labels into the preprocessing output dir.
        Keeps feature_store untouched so DataIngestion output is never modified.
        """
        try:
            os.makedirs(self.out_images_dir,exist_ok=True)
            os.makedirs(self.out_labels_dir,exist_ok=True)

            for img_file in os.listdir(self.src_images_dir):
                shutil.copy(
                    os.path.join(self.src_images_dir, img_file),
                    os.path.join(self.out_images_dir, img_file)
                )

            for lbl_file in os.listdir(self.src_labels_dir):
                shutil.copy(
                    os.path.join(self.src_labels_dir, lbl_file),
                    os.path.join(self.out_labels_dir, lbl_file)
                )               
            
            logger.info(f"Copied train split to:{self.data_preprocessing_config.data_preprocessing_dir}")

        except Exception as e:
            raise VehicleException(e,sys)
        
    def _update_data_yaml(self):
        """
        Rewrite data.yaml train path to point to the preprocessed train images dir.
        val and test paths are also resolved to absolute paths to prevent YOLO
        from misresolving them relative to data.yaml's directory.
        """
        try:
            with open(self.data_yaml_path, "r") as f:
                data = yaml.safe_load(f)

            feature_store = self.data_ingestion_artifact.feature_store_file_path

            # ← Fix: all paths must be absolute
            data["train"] = os.path.abspath(self.out_images_dir)
            data["val"]   = os.path.abspath(os.path.join(feature_store, "valid", "images"))
            data["test"]  = os.path.abspath(os.path.join(feature_store, "test", "images"))

            with open(self.data_yaml_path, "w") as f:
                yaml.dump(data, f)

            logger.info(f"Updated data.yaml paths:")
            logger.info(f"  train: {data['train']}")
            logger.info(f"  val: {data['val']}")
            logger.info(f"  test: {data['test']}")

        except Exception as e:
            raise VehicleException(e, sys)
        
    def _get_class_counts(self) -> dict:
        """Count bounding box instances per class in the output train labels dir."""
        try:
            class_counts = {}
            for label_file in os.listdir(self.out_labels_dir):
                label_path = os.path.join(self.out_labels_dir, label_file)
                with open(label_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        cls_id = int(float(line.split()[0]))  # ← fix here
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            return class_counts

        except Exception as e:
            raise VehicleException(e, sys)  
          
    def oversample_minority_classes(self) -> int:
        """
        Duplicate images that contain minority class instances.
        Minority = classes in the bottom minority_percentile by count.
        Operates on the output dir, not the original feature_store.

        Returns:
            Number of image-label pairs duplicated.
        """
        try:
            cfg = self.data_preprocessing_config
            class_counts = self._get_class_counts()

            threshold = np.percentile(list(class_counts.values()),
                                    cfg.minority_percentile)

            minority_classes = {cls_id for cls_id, cnt in class_counts.items() if cnt < threshold}

            logger.info(f"Minority class ID's (bottom {cfg.minority_percentile}%): {minority_classes}")

            duplicated = 0

            for label_file in os.listdir(self.out_labels_dir):
                label_path = os.path.join(self.out_labels_dir, label_file)
                with open(label_path, "r") as f:
                    lines = [l for l in f.readlines() if l.strip()]  # skip empty lines

                contains_minority = any(
                    int(float(line.split()[0])) in minority_classes  #  fix: float() first
                    for line in lines
                )

                if not contains_minority:
                    continue

                image_file = label_file.replace(".txt", ".jpg")
                src_img = os.path.join(self.out_images_dir, image_file)

                if not os.path.exists(src_img):
                    continue

                for i in range(cfg.duplicate_count):
                    new_img_name = image_file.replace(".jpg", f"_dup{i}.jpg")
                    new_lbl_name = label_file.replace(".txt", f"_dup{i}.txt")

                    shutil.copy(src_img, os.path.join(self.out_images_dir, new_img_name))
                    shutil.copy(label_path, os.path.join(self.out_labels_dir, new_lbl_name))
                    duplicated += 1

            logger.info(f"Oversampling complete. Duplicated {duplicated} minority class samples.")
            return duplicated

        except Exception as e:
            raise VehicleException(e, sys)
    
    
    def _get_fog_border_color(self, img: np.ndarray) -> tuple:
        """
        Sample the average color of the top 10% of the image (sky/fog region).
        Used as CoarseDropout fill so patches blend with the foggy background.
        """
        h = img.shape[0]
        sky_region = img[:max(1, h // 10), :, :]
        mean_color = sky_region.mean(axis=(0, 1))
        return tuple(int(c) for c in mean_color)
    
    def _build_augmentation_pipeline(self,
                                     fog_color: tuple = (180,180,180)) -> A.Compose:
        """
        Build the albumentations pipeline for fog-aware augmentation.
        Fog color is sampled per-image so CoarseDropout patches blend naturally.
        """

        r,g,b = fog_color

        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                size=(640, 640),
                scale=(0.80, 1.0),
                ratio=(1.0, 1.0),
                p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            # gamma_limit=(90,110): val<1.0 → brighter, val>1.0 → darker
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            # CLAHE: local contrast enhancement — helps recover details in fog
            # clip_limit prevents over-enhancement of noise
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            # Small hue/sat/val shifts — fog is grayish so keep shifts subtle
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
            # Simulates camera motion blur in foggy conditions
            A.GaussianBlur(blur_limit=(3, 3), p=0.3),
            # Hides small patches filled with fog-sky color
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill=(r, g, b),
                p=0.2
            ),
        ], bbox_params=A.BboxParams(
            format="yolo",
            label_fields=['class_labels'],
            min_visibility=0.3,
        ))

    def apply_augmentations(self) -> int:
        """
        Generate n_augments augmented copies of every image+label pair
        in the output train dir (after oversampling).

        Returns:
            Number of augmented image-label pairs generated.
        """
        try:
            cfg = self.data_preprocessing_config
            augmented = 0
            skipped = 0

            for img_file in os.listdir(self.out_images_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(self.out_images_dir, img_file)
                lbl_path = os.path.join(
                    self.out_labels_dir,
                    os.path.splitext(img_file)[0] + '.txt'
                )

                if not os.path.exists(lbl_path):
                    skipped += 1
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                fog_color = self._get_fog_border_color(img)
                pipeline = self._build_augmentation_pipeline(fog_color=fog_color)

                with open(lbl_path) as f:
                    lines = [l for l in f.readlines() if l.strip()]  # ← skip empty lines

                bboxes, class_labels = [], []
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    class_labels.append(int(parts[0]))   # ← already safe via map(float)
                    bboxes.append(parts[1:5])

                if not bboxes:   # ← skip images with no valid boxes
                    skipped += 1
                    continue

                for aug_idx in range(cfg.n_augments):
                    try:
                        result = pipeline(image=img, bboxes=bboxes, class_labels=class_labels)

                        if len(result['bboxes']) == 0:
                            continue

                        aug_img = cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR)
                        base_name = os.path.splitext(img_file)[0]

                        cv2.imwrite(
                            os.path.join(self.out_images_dir, f"{base_name}_aug{aug_idx}.jpg"),
                            aug_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95]
                        )

                        with open(
                            os.path.join(self.out_labels_dir, f"{base_name}_aug{aug_idx}.txt"),
                            'w'
                        ) as f:
                            for cls, bbox in zip(result['class_labels'], result['bboxes']):
                                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")

                        augmented += 1

                    except Exception:
                        skipped += 1

            logger.info(f"Augmentation complete. Generated: {augmented} | Skipped: {skipped}")
            return augmented

        except Exception as e:
            raise VehicleException(e, sys)

    def _is_artifact_available(self) -> bool:
        images_dir = self.data_preprocessing_config.preprocessed_train_images_dir
        if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
            logger.info("Preprocessed data already exists — skipping preprocessing.")
            return True
        return False
        
    def initiate_data_preprocessing(self) -> DataPreprocessingArtifact:
        logger.info("Initiating data preprocessing")
        try:
            out_images_dir = self.data_preprocessing_config.preprocessed_train_images_dir
            out_labels_dir = self.data_preprocessing_config.preprocessed_train_labels_dir

            if os.path.exists(out_images_dir) and len(os.listdir(out_images_dir)) > 0:
                logger.info(f"Preprocessed data already exists at: {out_images_dir} — skipping preprocessing.")
                return DataPreprocessingArtifact(
                    preprocessed_train_images_dir=out_images_dir,
                    preprocessed_train_labels_dir=out_labels_dir,
                    duplicated_count=0,
                    augmented_count=0,
                )

            logger.info("No preprocessed data found — proceeding with data preprocessing.")
            self._copy_train_split()
            self._update_data_yaml()
            duplicated_count = self.oversample_minority_classes()
            augmented_count = self.apply_augmentations()

            artifact = DataPreprocessingArtifact(
                preprocessed_train_images_dir=out_images_dir,
                preprocessed_train_labels_dir=out_labels_dir,
                duplicated_count=duplicated_count,
                augmented_count=augmented_count,
            )
            logger.info(f"Data preprocessing artifact: {artifact}")
            return artifact

        except Exception as e:
            raise VehicleException(e, sys)
