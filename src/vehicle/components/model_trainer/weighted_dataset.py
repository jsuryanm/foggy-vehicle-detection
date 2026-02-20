import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build

class YOLOWeightedDataset(YOLODataset):
    """
    Improved weighted sampler for YOLO26 foggy vehicle detection.

    Improvements over v1:
    - MAX aggregation (not mean) -> rare class images get sampling priority
    - Temperature scaling on weights -> controls sharpness of sampling distribution
      temperature > 1.0 = smoother (less aggressive), < 1.0 = sharper (more aggressive)

    Note: defined at module top-level (not inside a function) so it is picklable
    on Windows where multiprocessing uses 'spawn' instead of 'fork'.
    """

    default_temperature: float = 1.5  # overwritten by patch_yolo_with_weighted_dataset()

    def __init__(self, *args, mode="train", **kwargs):
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.temperature = self.__class__.default_temperature  # ← reads class-level value

        self.count_instances()

        # Inverse-frequency weights with temperature scaling
        class_weights = np.sum(self.counts) / (self.counts + 1e-6)
        class_weights = class_weights ** (1.0 / self.temperature)
        self.class_weights = class_weights

        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """Count instances per class across all labels."""
        self.counts = np.zeros(len(self.data["names"]), dtype=np.float32)

        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Assign each image a weight based on the rarest class it contains.
        MAX aggregation ensures images with rare classes are always prioritised.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            if cls.size == 0:
                weights.append(1.0)
                continue

            weight = np.max(self.class_weights[cls])
            weights.append(float(weight))

        return weights

    def calculate_probabilities(self):
        """Normalize weights into sampling probabilities."""
        total = sum(self.weights)
        return [w / total for w in self.weights]

    def __getitem__(self, index):
        """During training, sample by probability; during val/test use normal index."""
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))

        index = np.random.choice(len(self.labels), p=self.probabilities)
        return self.transforms(self.get_image_and_label(index))


def patch_yolo_with_weighted_dataset(temperature: float = 1.5):
    """
    Monkey-patch ultralytics to use YOLOWeightedDataset instead of YOLODataset.
    Must be called BEFORE model.train().

    Args:
        temperature: Controls sampling sharpness. > 1.0 = smoother, < 1.0 = sharper.
    """
    # Set temperature as a class-level default so no inner/local class is needed.
    # Inner classes defined inside functions are NOT picklable on Windows (spawn),
    # which breaks PyTorch multiprocessing DataLoader workers.
    YOLOWeightedDataset.default_temperature = temperature
    build.YOLODataset = YOLOWeightedDataset