# 🚗 Foggy Vehicle Detection System

A vehicle detection system built with **YOLOv26** that reliably detects vehicles in **foggy and low-visibility** environments.

---

## 📌 Overview

This project trains an object detection model capable of identifying vehicles (cars, trucks, buses, etc.) under adverse weather conditions — specifically fog, haze, and low-contrast scenes.

**Key Highlights:**
- Fine-tuning YOLOv26m on fog vehicle dataset
- Fog-aware albumentations augmentation pipeline
- Weighted dataset sampler for class imbalance (MAX aggregation + temperature scaling)
- Minority class oversampling
- IoU and confidence threshold sweeps for optimal NMS
- Test Time Augmentation (TTA) for evaluation
- Full training pipeline: ingestion → validation → preprocessing → training → evaluation

---

## 🗂️ Project Structure

```
FOGGY-VEHICLE-DETECTION/
├── src/vehicle/
│   ├── components/
│   │   ├── data_ingestion.py        # Downloads & extracts dataset
│   │   ├── data_validation.py       # Validates dataset structure
│   │   ├── data_preprocessing.py    # Oversampling + augmentation
│   │   ├── model_trainer/
│   │   │   ├── trainer.py           # YOLO26 training logic
│   │   │   └── weighted_dataset.py  # Custom weighted sampler
│   │   └── model_evaluation.py      # Evaluation + IoU/conf sweeps
│   ├── constants/training_pipeline/
│   │   ├── constant.py              # Global constants
│   │   ├── model_trainer_constants.py
│   │   └── model_eval_constants.py
│   ├── entity/
│   │   ├── config_entity.py         # Dataclass configs
│   │   └── artifacts_entity.py      # Dataclass artifacts
│   ├── pipeline/
│   │   └── training_pipeline.py     # End-to-end pipeline runner
│   ├── logger/logger.py
│   ├── exceptions/exception.py
│   └── utils/main_utils.py
├── runs/                            # YOLO training outputs (weights, plots)
├── artifacts/                       # Generated pipeline artifacts
├── logs/                            # Runtime logs
├── research/                        # Notebooks & experiments
├── app.py                           # FastAPI backend
├── streamlit_app.py                 # Streamlit demo UI
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── setup.py
```

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/foggy-vehicle-detection.git
cd foggy-vehicle-detection
```

### 2. Create & Activate Environment
```bash
conda create -n myenv python=3.12 -y
conda activate myenv
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running the Training Pipeline

Run the full end-to-end pipeline (ingestion → validation → preprocessing → training → evaluation):

```bash
python src/vehicle/pipeline/training_pipeline.py
```

---

## 📊 Pipeline Components

| Component | Description |
|---|---|
| **Data Ingestion** | Downloads dataset from Google Drive, extracts ZIP |
| **Data Validation** | Checks for required splits (`train/`, `valid/`, `test/`, `data.yaml`) |
| **Data Preprocessing** | Copies train split, we oversample minority classes (2×) and apply fog-aware albumentations |
| **Model Trainer** | Trains YOLOv26 with MuSGD optimizer, with weighted sampling, and fog augmentations |
| **Model Evaluation** | IoU sweep, confidence sweep, per-class AP, TTA comparison |

---

## 🧪 Augmentation Strategy

The preprocessing pipeline applies fog-aware augmentations using **Albumentations**:

- `HorizontalFlip` — basic geometric diversity  
- `RandomResizedCrop` — scale invariance  
- `RandomBrightnessContrast` — handles varying fog density  
- `CLAHE` — local contrast enhancement to recover details in fog  
- `HueSaturationValue` — subtle color shifts (fog is grayish)  
- `GaussianBlur` — simulates camera motion blur in fog  
- `CoarseDropout` — patches filled with fog-sky color (sampled from top 10% of image)

---

## 📈 Training Configuration

| Parameter | Value |
|---|---|
| Model | `yolo26m.pt` |
| Epochs | 150 |
| Batch Size | 64 |
| Image Size | 640 |
| Optimizer | MuSGD |
| LR | 0.01 (cosine decay) |
| Warmup Epochs | 5 |
| Patience | 40 |
| AMP | ✅ |
| Cache | disk |

---

## 🔍 Evaluation

Evaluation runs a 3-stage sweep on the validation set to find optimal inference settings:

1. **Standard vs TTA** — compares baseline vs test-time augmentation
2. **IoU Sweep** `[0.45 → 0.70]` — finds best NMS threshold by mAP50-95
3. **Confidence Sweep** `[0.10 → 0.50]` — finds best confidence threshold by F1

Final evaluation runs on the **test set** with TTA + best IoU + best confidence.

---

## 📦 Dataset

The dataset is sourced from **Roboflow** (foggy car detection dataset) and downloaded automatically during the data ingestion step. It includes:

- `train/` — training images + YOLO format labels  
- `valid/` — validation split  
- `test/` — held-out test split  
- `data.yaml` — class names and split paths

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| YOLOv26 (Ultralytics) | Object detection |
| Albumentations | Fog-aware augmentation |
| OpenCV | Image processing |
| PyTorch | Deep learning backend |
| Roboflow | Dataset management |
| Streamlit | Demo UI |
| FastAPI | REST API |
| Docker | Containerization |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---
