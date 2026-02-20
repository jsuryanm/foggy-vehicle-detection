# ─────────────────────────────────────────────
# Optimizer Settings
# ─────────────────────────────────────────────
MODEL_TRAINER_OPTIMIZER: str = "MuSGD"           # YOLO26's native optimizer
MODEL_TRAINER_LR0: float = 0.01                  # SGD-range LR (not AdamW-range 0.001)
MODEL_TRAINER_LRF: float = 0.1                   # Final LR fraction
MODEL_TRAINER_MOMENTUM: float = 0.937
MODEL_TRAINER_WEIGHT_DECAY: float = 5e-4
MODEL_TRAINER_COS_LR: bool = True                # Cosine LR scheduler

# ─────────────────────────────────────────────
# Warmup Settings
# ─────────────────────────────────────────────
MODEL_TRAINER_WARMUP_EPOCHS: float = 5.0
MODEL_TRAINER_WARMUP_BIAS_LR: float = 0.05
MODEL_TRAINER_WARMUP_MOMENTUM: float = 0.8

# ─────────────────────────────────────────────
# Fog-Aware Augmentation Settings 
# ─────────────────────────────────────────────
MODEL_TRAINER_MOSAIC: float = 1.0
MODEL_TRAINER_CLOSE_MOSAIC: int = 15             # Updated: was 20
MODEL_TRAINER_MIXUP: float = 0.1                 # New: added mixup
MODEL_TRAINER_COPY_PASTE: float = 0.2            # Updated: was 0.1
MODEL_TRAINER_HSV_H: float = 0.015
MODEL_TRAINER_HSV_S: float = 0.5                 # Updated: was 0.4
MODEL_TRAINER_HSV_V: float = 0.5                 # Updated: was 0.4
MODEL_TRAINER_SCALE: float = 0.5
MODEL_TRAINER_TRANSLATE: float = 0.1
MODEL_TRAINER_FLIPLR: float = 0.5               # New: explicit horizontal flip
MODEL_TRAINER_DEGREES: float = 5.0              # New: rotation augmentation
MODEL_TRAINER_ERASING: float = 0.2              # New: random erasing

# ─────────────────────────────────────────────
# Loss Weights 
# ─────────────────────────────────────────────
MODEL_TRAINER_BOX_LOSS: float = 7.5             # Updated: was 9.0
MODEL_TRAINER_CLS_LOSS: float = 0.5


# ─────────────────────────────────────────────
# Performance & Stability Settings
# ─────────────────────────────────────────────
MODEL_TRAINER_CACHE: str = "ram"
MODEL_TRAINER_AMP: bool = True
MODEL_TRAINER_COMPILE: bool = False              # Disabled: can cause issues on some CUDA versions
MODEL_TRAINER_PRETRAINED: bool = True
MODEL_TRAINER_FREEZE: int = 0                   # Train all layers (good for fog domain adaptation)
MODEL_TRAINER_WORKERS: int = 2
MODEL_TRAINER_PATIENCE: int = 40                 

# ─────────────────────────────────────────────
# Output Settings
# ─────────────────────────────────────────────
MODEL_TRAINER_RUN_NAME: str = "fog_vehicle_musgd"  
MODEL_TRAINER_EXIST_OK: bool = True

# ─────────────────────────────────────────────
# Weighted Dataset Settings
# ─────────────────────────────────────────────
WEIGHTED_DATASET_TEMPERATURE: float = 1.5       # > 1 = smoother, < 1 = sharper sampling

