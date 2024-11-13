"""Distance metric implementations."""

import numpy as np
from core.base import DistanceMetric

class EuclideanDistance(DistanceMetric):
    """Euclidean distance metric."""
    
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(np.sum((x - y) ** 2))
    
    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = x - y
        norm = np.sqrt(np.sum(diff ** 2))
        return diff / (norm + 1e-10) 