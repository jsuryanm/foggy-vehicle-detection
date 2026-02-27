import os
import shutil
import tempfile
import json
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.vehicle.components.inference import VehicleDetector
from src.vehicle.entity.config_entity import ModelTrainerConfig


cfg = ModelTrainerConfig()

MODEL_PATH = os.path.abspath(
    os.path.join(cfg.model_trainer_dir, "weights", cfg.trained_model_file_path)
)

_detector = None


def get_detector():
    global _detector

    if _detector is None:
        print("Loading YOLO model (Lazy Initialization)...")
        _detector = VehicleDetector(MODEL_PATH)

    return _detector



app = FastAPI(title="Fog Vehicle Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "API running. model is not loaded yet)"}




@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.6)
):
    detector = get_detector()  # Lazy load here

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    annotated, counts, inference_time = detector.predict_image(img, conf, iou)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp.name, annotated)

    return JSONResponse(content={
        "image_path": tmp.name,
        "counts": counts,
        "inference_time": inference_time
    })


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.6)
):
    detector = get_detector()  # Lazy load here

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    with open(temp_input.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    output_path, counts, inference_time = detector.predict_video(
        temp_input.name,
        temp_output.name,
        conf,
        iou
    )

    response = FileResponse(output_path, media_type="video/mp4")
    response.headers["X-Counts"] = json.dumps(counts)
    response.headers["X-Inference-Time"] = str(inference_time)

    return response