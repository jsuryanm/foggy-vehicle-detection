import os 
import sys 
from src.vehicle.logger.logger import logger 
from src.vehicle.exceptions.exception import VehicleException
from src.vehicle.entity.config_entity import DataValidationConfig
from src.vehicle.entity.artifacts_entity import (DataIngestionArtifact,
                                                 DataValidationArtifact)

class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        
        except Exception as e:
            raise VehicleException(e,sys)
        
    def validate_all_files(self) -> bool:
        try:
            validation_status = True

            all_files = os.listdir(
                self.data_ingestion_artifact.feature_store_file_path
            )

            for required_file in self.data_validation_config.required_file_list:
                if required_file not in all_files:
                    validation_status = False
                    break

            # Create ONLY the directory
            os.makedirs(
                self.data_validation_config.data_validation_dir,
                exist_ok=True
            )

            # Now create/write the status file
            with open(
                self.data_validation_config.valid_status_file_dir,
                "w"
            ) as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise VehicleException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        logger.info("Initiated data validation")
        try:
            status = self.validate_all_files()
            data_validation_artifact = DataValidationArtifact(validation_status=status)
            logger.info(f"Data validation artifact:{data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise VehicleException(e,sys)

