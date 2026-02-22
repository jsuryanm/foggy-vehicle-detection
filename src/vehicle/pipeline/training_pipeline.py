import os 
import sys 

from src.vehicle.logger.logger import logger 
from src.vehicle.exceptions.exception import VehicleException
from src.vehicle.components.data_ingestion import DataIngestion
from src.vehicle.components.data_validation import DataValidation
from src.vehicle.components.data_preprocessing import DataPreprocessing
from src.vehicle.components.model_trainer.trainer import ModelTrainer
from src.vehicle.components.model_evaluation import ModelEvaluation

from src.vehicle.entity.config_entity import (DataIngestionConfig,
                                              DataValidationConfig,
                                              DataPreprocessingConfig,
                                              ModelTrainerConfig,
                                              ModelEvaluationConfig)

from src.vehicle.entity.artifacts_entity import (DataIngestionArtifact,
                                                 DataValidationArtifact,
                                                 DataPreprocessingArtifact,
                                                 ModelTrainerArtifact,
                                                 ModelEvaluationArtifact)


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()    
        self.model_trainer_config = ModelTrainerConfig()
        self.model_eval_config = ModelEvaluationConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Initiated data ingestion component from training pipeline")

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logger.info("Exited Data Ingestion component from training pipeline")
            return data_ingestion_artifact
        
        except Exception as e:
            raise VehicleException(e,sys)
    
    def start_data_validation(self,
                              data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logger.info("Initiated data validation component from training pipeline")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config)
            
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info("Completed data validation component from training pipeline")

            return data_validation_artifact
        except Exception as e:
            raise VehicleException(e,sys)
        
    def start_data_preprocessing(self,
                                 data_ingestion_artifact: DataIngestionArtifact) -> DataPreprocessingArtifact:
        try:
            logger.info("Initiated data preprocessing component from training pipeline")
            data_preprocessing = DataPreprocessing(data_ingestion_artifact=data_ingestion_artifact,
                                                   data_preprocessing_config=self.data_preprocessing_config)
            
            data_preprocessing_artifact = data_preprocessing.initiate_data_preprocessing()
            logger.info("Completed data preprocessing component from training pipeline")
            return data_preprocessing_artifact
        except Exception as e:
            raise VehicleException(e,sys)
    
    def start_model_trainer(self,
                            data_ingestion_artifact: DataIngestionArtifact,
                            data_validation_artifact: DataValidationArtifact) -> ModelTrainerArtifact:
        try:
            if not data_validation_artifact.validation_status:
                raise Exception("Data validation failed. Cannot proceed with training.")

            model_trainer = ModelTrainer(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info("Exited model trainer component from training pipeline")
            return model_trainer_artifact

        except Exception as e:
            raise VehicleException(e, sys)
    
    def start_model_eval(self,
                         data_ingestion_artifact: DataIngestionArtifact,
                         model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            logger.info("Initiated model evaluation component from training pipeline")

            model_eval = ModelEvaluation(model_trainer_artifact=model_trainer_artifact,
                                         data_ingestion_artifact=data_ingestion_artifact,
                                         model_evaluation_config=ModelEvaluationConfig())
            
            model_eval_artifact = model_eval.initiate_model_evaluation()
            logger.info("Completed model evaluation component from training pipeline")
            return model_eval_artifact

        except Exception as e:
            raise VehicleException(e, sys)
        
        
    def run_pipeline(self) -> None:
        try: 
            logger.info("Running all components from training pipeline")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_preprocessing_artifact = self.start_data_preprocessing(data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact = self.start_model_trainer(data_ingestion_artifact=data_ingestion_artifact,
                                                              data_validation_artifact=data_validation_artifact)
            model_eval_artifact = self.start_model_eval(data_ingestion_artifact=data_ingestion_artifact,
                                                        model_trainer_artifact=model_trainer_artifact)

        except Exception as e:
            raise VehicleException(e,sys)
        
if __name__ == "__main__":
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()