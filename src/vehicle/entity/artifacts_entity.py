from dataclasses import dataclass 

@dataclass
class DataIngestionArtifact:
    data_zip_file_path: str 
    feature_store_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool

@dataclass 
class DataPreprocessingArtifact:
    preprocessed_train_images_dir: str
    preprocessed_train_labels_dir: str 
    duplicated_count: int 
    augmented_count: int

@dataclass 
class ModelTrainerArtifact:
    trained_model_path: str