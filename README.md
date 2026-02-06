# Foggy Vehicle Detection System

![Project Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-red)

## 🎯 Project Overview

An end-to-end deep learning system for real-time vehicle detection in foggy and low-visibility conditions, designed to improve driver safety in adverse weather scenarios. This project implements YOLOv8 for object detection, trained on the Foggy Cityscapes Dataset.

### Key Features
- **Real-time vehicle detection** optimized for foggy conditions
- **Complete ML pipeline** from data preparation to deployment
- **Production-ready code** with modular architecture
- **Comprehensive evaluation** with multiple metrics
- **Docker containerization** for reproducible deployment
- **REST API** for easy integration

---

## 📋 Table of Contents
- [Project Status](#-project-status)
- [Technical Stack](#-technical-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Development Phases](#-development-phases)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Project Status

### ✅ Completed Phases

#### Phase 1: Environment Setup & Project Structure ✓
- [x] Local development environment configuration
- [x] Project directory structure created
- [x] Dependencies defined in `requirements.txt`
- [x] Git repository initialized
- [x] Configuration management setup

**Deliverables:**
- Complete project structure with organized directories
- `requirements.txt` with all necessary dependencies
- `config/config.yaml` for centralized configuration
- `.gitignore` configured for Python ML projects

### 🔄 Current Phase

#### Phase 2: Data Pipeline Development (IN PROGRESS)
- [ ] Download Foggy Cityscapes dataset from Roboflow
- [ ] Implement data exploration scripts
- [ ] Create visualization utilities
- [ ] Implement data augmentation pipeline
- [ ] Build custom data loaders

### 📅 Upcoming Phases

#### Phase 3: Model Architecture & Configuration
- [ ] Configure YOLOv8 model
- [ ] Set up transfer learning pipeline
- [ ] Define custom model configurations

#### Phase 4: Training Pipeline
- [ ] Implement training script
- [ ] Set up experiment tracking
- [ ] Configure hyperparameter optimization

#### Phase 5: Evaluation Framework
- [ ] Build comprehensive evaluation metrics
- [ ] Create visualization tools
- [ ] Generate performance reports

#### Phase 6: Inference & Testing
- [ ] Single image inference
- [ ] Batch inference
- [ ] Video processing pipeline

#### Phase 7: Model Optimization
- [ ] Hyperparameter tuning
- [ ] Model export (ONNX, TorchScript)
- [ ] Speed optimization

#### Phase 8: API Development
- [ ] FastAPI REST API
- [ ] Endpoint testing
- [ ] API documentation

#### Phase 9: Containerization
- [ ] Docker container creation
- [ ] Local container testing

#### Phase 10: AWS Deployment (FINAL)
- [ ] EC2 instance setup
- [ ] Model deployment
- [ ] CI/CD pipeline
- [ ] Monitoring and logging

---

## 🛠️ Technical Stack

### Core Technologies
- **Deep Learning Framework:** PyTorch 2.10
- **Object Detection:** Ultralytics YOLOv8,YOLOv26
- **Computer Vision:** OpenCV
- **Data Science:** NumPy, Pandas, Matplotlib

### Development Tools
- **Environment:** Python 3.19+, Conda/venv
- **Version Control:** Git
- **API Framework:** FastAPI (upcoming)
- **Containerization:** Docker (upcoming)
- **Cloud Platform:** AWS EC2 (final deployment)

### Dataset
- **Source:** [Foggy Cityscapes Dataset](https://app.roboflow.com/js-2wbtl/foggy-cityscapes-dataset-drkvl/1) via Roboflow
- **Format:** YOLO format (images + annotations)
- **Focus:** Vehicle detection in various fog densities
---

## 💻 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/foggy-vehicle-detection.git
cd foggy-vehicle-detection
```

2. **Create virtual environment**
```bash
# Using conda (recommended)
conda create -n foggy-detection python=3.8
conda activate foggy-detection

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
```

### Dependencies Overview
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLOv8 implementation
- **OpenCV**: Image processing
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **PyYAML**: Configuration management
- **Roboflow**: Dataset management

---

## 📊 Dataset

### Foggy Cityscapes Dataset

The project uses the **Foggy Cityscapes Dataset** from Roboflow, which contains:
- Urban driving scenes with varying fog densities
- Multiple vehicle classes (cars, trucks, buses, etc.)
- High-quality bounding box annotations in YOLO format
- Train/validation/test splits

### Downloading the Dataset

**Method 1: Using Roboflow Python API** (Recommended)
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("js-2wbtl").project("foggy-cityscapes-dataset-drkvl")
dataset = project.version(1).download("yolov8")
```

**Method 2: Manual Download**
1. Visit [Foggy Cityscapes Dataset](https://app.roboflow.com/js-ozptv/foggy-car-ofcf4/1)
2. Download in YOLOv8 format
3. Extract to `data/raw/`

### Dataset Structure (Expected)
```
data/raw/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## 🎮 Usage

### Data Exploration (Coming Soon)
```bash
# Run data exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb


```

### Training (Coming Soon)
```bash
# Basic training
python src/training/train.py --config config/config.yaml

# Training with custom parameters
python src/training/train.py \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640
```

### Inference (Coming Soon)
```bash
# Single image
python src/inference/predict.py --source path/to/image.jpg

# Video
python src/inference/video_inference.py --source path/to/video.mp4

# Directory
python src/inference/predict.py --source path/to/images/
```

### API Server (Coming Soon)
```bash
# Start API server
uvicorn src.api.main:app --reload

# Test endpoint
curl -X POST "http://localhost:8000/predict" \
     -F "file=@test_image.jpg"
```

---

## 🔬 Development Phases

### Development Approach
This project follows a **local-first development strategy**:
1. ✅ **Build locally** - Complete end-to-end system on local machine
2. ✅ **Test thoroughly** - Ensure everything works before deployment
3. 📅 **Deploy to AWS** - Final phase after local validation

### Phase Details

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Environment Setup | ✅ Complete | Project structure, dependencies, configs |
| 2. Data Pipeline | 🔄 In Progress | Download, EDA, augmentation, loaders |
| 3. Model Config | 📅 Pending | YOLO setup, transfer learning |
| 4. Training | 📅 Pending | Training loop, checkpointing, logging |
| 5. Evaluation | 📅 Pending | Metrics, visualization, reports |
| 6. Inference | 📅 Pending | Image/video processing pipeline |
| 7. Optimization | 📅 Pending | Hyperparameter tuning, export |
| 8. API Development | 📅 Pending | REST API with FastAPI |
| 9. Containerization | 📅 Pending | Docker setup and testing |
| 10. AWS Deployment | 📅 Pending | EC2 deployment, CI/CD |

---

## 📈 Performance Metrics

### Target Metrics
- **mAP@0.5**: > 0.75
- **mAP@0.5:0.95**: > 0.60
- **Inference Speed**: > 30 FPS (on target hardware)
- **Precision**: > 0.80
- **Recall**: > 0.75

*(Actual results will be updated after training)*

---



### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features
- Update documentation as needed

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for the YOLO implementations
- **Roboflow** for the Foggy Cityscapes Dataset
- **PyTorch** team for the deep learning framework

---

## 📧 Contact

**Project Maintainer:** [Jayasuryan Mutyala]
- GitHub: [@jsuryanm](https://github.com/jsuryanm)

---

## 🗺️ Roadmap

### Short-term (Current Sprint)
- [ ] Complete data exploration and analysis
- [ ] Implement data augmentation pipeline
- [ ] Set up baseline YOLO model training

### Medium-term (Next Month)
- [ ] Achieve target mAP metrics
- [ ] Optimize inference speed
- [ ] Deploy local API server

### Long-term (Future)
- [ ] AWS EC2 deployment
- [ ] Real-time video streaming support
- [ ] Mobile app integration
- [ ] Multi-weather condition support

---

## 📚 Resources

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Roboflow Documentation](https://docs.roboflow.com/)

### Related Papers
- YOLO: Real-Time Object Detection
- Foggy Cityscapes: Semantic Foggy Scene Understanding

---

**Last Updated:** February 2026  
**Version:** 0.1.0 (Development)