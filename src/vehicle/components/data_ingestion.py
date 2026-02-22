import os 
import sys 
import zipfile 
import gdown 

from src.vehicle.logger.logger import logger 
from src.vehicle.exceptions.exception import VehicleException
from src.vehicle.entity.config_entity import DataIngestionConfig
from src.vehicle.entity.artifacts_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise VehicleException(e,sys)
    
    def download_data(self) -> str:
        '''
        Fetch data from url 
        '''

        try:
            dataset_url = self.data_ingestion_config.data_download_url
            zip_download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_download_dir,exist_ok=True)
            data_file_name = "data.zip"
            zip_file_path = os.path.join(zip_download_dir,data_file_name)
            logger.info(f"Downloading data from {dataset_url} into {zip_file_path}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id,zip_file_path)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_file_path}")
            return zip_file_path
        
        except Exception as e:
            raise VehicleException(e,sys)
        
    def extract_zip_file(self,zip_file_path: str) -> str:
        '''
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        '''

        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path,exist_ok=True)
            with zipfile.ZipFile(zip_file_path,"r") as zip_ref:
                zip_ref.extractall(feature_store_path)
            logger.info(f"Extract zip file path: {zip_file_path} into dir: {feature_store_path}")
            return feature_store_path
        
        except Exception as e:
            raise VehicleException(e,sys)
    
    def _is_artifact_available(self) -> bool:
        """
        Skip data ingestion if the feature store already exists and is non-empty.
        We only check feature_store_file_path since that's what the config tracks.
        """
        feature_store = self.data_ingestion_config.feature_store_file_path

        if os.path.exists(feature_store) and len(os.listdir(feature_store)) > 0:
            logger.info(f"Feature store already exists at: {feature_store} — skipping download & extraction.")
            return True

        logger.info("Feature store not found or empty — proceeding with data ingestion.")
        return False

    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Initiating data ingestion")
        try:
            if self._is_artifact_available():
                # Reconstruct the zip path the same way download_data() does
                # so the artifact is consistent even when skipped
                zip_file_path = os.path.join(
                    self.data_ingestion_config.data_ingestion_dir, "data.zip"
                )
                return DataIngestionArtifact(
                    data_zip_file_path=zip_file_path,
                    feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                )

            zip_file_path = self.download_data()
            feature_store_path = self.extract_zip_file(zip_file_path)

            artifact = DataIngestionArtifact(
                data_zip_file_path=zip_file_path,
                feature_store_file_path=feature_store_path,
            )
            logger.info(f"Data ingestion artifact: {artifact}")
            return artifact

        except Exception as e:
            raise VehicleException(e, sys)

# if __name__ == "__main__":
#     data_ingestion_obj = DataIngestion()
#     data_ingestion_obj.initiate_data_ingestion()