"""Signed distance calculations."""

import numpy as np
from typing import Dict
from .metrics import EuclideanDistance
from core.base import DistanceMetric

class SignedDistanceCalculator:
    """Calculator for signed distances in high dimensions."""
    
    def __init__(self, metric: DistanceMetric = EuclideanDistance()):
        self.metric = metric
        
    def calculate_delta_p0(self, dims: np.ndarray) -> np.ndarray:
        """Calculate signed distance for zero vector."""
        return -1 / np.sqrt(dims)
    
    def calculate_delta_p1(self, dims: np.ndarray) -> np.ndarray:
        """Calculate signed distance for all-ones vector."""
        return (dims - 1) / np.sqrt(dims)
    
    def analyze_distance_scaling(self, 
                               max_dim: int = 100,
                               n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Analyze how distances scale with dimension."""
        dims = np.arange(1, max_dim + 1)
        mean_distances = np.zeros(max_dim)
        std_distances = np.zeros(max_dim)
        
        for i, d in enumerate(dims):
            distances = []
            for _ in range(n_samples):
                x = np.random.randn(d)
                y = np.random.randn(d)
                distances.append(self.metric.calculate(x, y))
            distances = np.array(distances)
            mean_distances[i] = np.mean(distances)
            std_distances[i] = np.std(distances)
            
        return {
            'dimensions': dims,
            'mean_distances': mean_distances,
            'std_distances': std_distances
        } 