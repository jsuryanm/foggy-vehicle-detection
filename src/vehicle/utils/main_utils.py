import os 
import sys 
import yaml 

from src.vehicle.exceptions.exception import VehicleException
from src.vehicle.logger.logger import logger 


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            logger.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise VehicleException(e,sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,"w") as file:
            yaml.dump(content,file)
            logger.info("Successfully written to yaml file")
    
    except Exception as e:
        raise VehicleException(e,sys)