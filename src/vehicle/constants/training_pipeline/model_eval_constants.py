# ─────────────────────────────────────────────
# Model Evaluation Constants
# ─────────────────────────────────────────────

# Default thresholds (for initial standard/TTA eval)
EVALUATION_CONF_DEFAULT: float = 0.001     # low conf → let NMS filter
EVALUATION_IOU_DEFAULT: float = 0.6

# IoU sweep range for NMS tuning
EVALUATION_IOU_SWEEP: list = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

# Confidence sweep range for F1 optimization
EVALUATION_CONF_SWEEP: list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

# Evaluation split
EVALUATION_SPLIT: str = "test"

# Output
EVALUATION_DIR_NAME: str = "model_evaluation"
EVALUATION_REPORT_FILE: str = "evaluation_report.json"
EVALUATION_PLOTS_DIR: str = "plots"