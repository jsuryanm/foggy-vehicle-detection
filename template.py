import os 
from pathlib import Path 
import logging 

logging.basicConfig(level=logging.INFO,format="[%(asctime)s]: %(message)s")

project_name = "vehicle"

list_of_files = [
    ".github/workflows/.gitkeep",
    "data/.gitkeep",
    "setup.py",
    "requirements.txt",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/constants/training_pipeline/__init__.py",
    f"src/{project_name}/constants/application.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifacts_entity.py",
    f"src/{project_name}/exceptions/__init__.py",
    f"src/{project_name}/exceptions/exception.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/logger/logger.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    "research/trials.ipynb",
    "app.py",
    "streamlit_app.py",
    "Dockerfile.api",
    "Dockerfile.ui",
    "docker-compose.local.yml",
    "docker-compose.yml"
] 

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file {filename}")

    if (not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(filepath,"w") as f:
            pass 
            logging.info(f"Creating empty file: {filename}")

    else:
        logging.info(f"{filename} is already created") 
