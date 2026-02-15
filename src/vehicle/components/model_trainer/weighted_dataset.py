import numpy as np
from ultralytics.data.dataset import YOLODataset

class YOLOWeighedDataset(YOLODataset):
    """
    Custom YOLO Dataset with Weighted Sampling
    Helps address class imbalance by oversampling rare classes.
    """

    def __init__(self, *args, data = None, mode = "train", **kwargs):
        super().__init__(*args, data=data, mode=mode, **kwargs)
        self.train_mode = mode == "train"