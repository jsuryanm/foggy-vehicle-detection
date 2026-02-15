import os 
import sys 

from src.vehicle.logger.logger import logger 
from src.vehicle.exceptions.exception import VehicleException
from src.vehicle.components.data_ingestion import DataIngestion
from src.vehicle.components.data_validation import DataValidation

from src.vehicle.entity.config_entity import (DataIngestionConfig,
                                              DataValidationConfig)
from src.vehicle.entity.artifacts_entity import (DataIngestionArtifact,
                                                 DataValidationArtifact)


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        
    
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
        
    def run_pipeline(self) -> None:
        try: 
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        except Exception as e:
            raise VehicleException(e,sys)
        
if __name__ == "__main__":
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()