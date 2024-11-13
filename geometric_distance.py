"""
Geometric Distance Analysis Library
=================================

A comprehensive library for analyzing geometric distances in high dimensions:

- Distance calculations and metrics
- Statistical analysis of distance distributions
- Visualization tools and plotting utilities
- Dimensionality analysis
- Advanced geometric computations
- Extensible interfaces for custom metrics

The implementation follows clean design principles with modular components.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""
    
    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance between two points.
        
        Args:
            x: First point
            y: Second point
            
        Returns:
            Distance value
        """
        pass
    
    @abstractmethod
    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate gradient of distance metric.
        
        Args:
            x: First point
            y: Second point
            
        Returns:
            Gradient vector
        """
        pass

class EuclideanDistance(DistanceMetric):
    """Euclidean distance metric."""
    
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(np.sum((x - y) ** 2))
    
    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = x - y
        norm = np.sqrt(np.sum(diff ** 2))
        return diff / (norm + 1e-10)

class SignedDistanceCalculator:
    """Calculator for signed distances in high dimensions."""
    
    def __init__(self, metric: DistanceMetric = EuclideanDistance()):
        self.metric = metric
        
    def calculate_delta_p0(self, dims: np.ndarray) -> np.ndarray:
        """Calculate signed distance for zero vector.
        
        Args:
            dims: Array of dimensions
            
        Returns:
            Array of signed distances
        """
        return -1 / np.sqrt(dims)
    
    def calculate_delta_p1(self, dims: np.ndarray) -> np.ndarray:
        """Calculate signed distance for all-ones vector.
        
        Args:
            dims: Array of dimensions
            
        Returns:
            Array of signed distances
        """
        return (dims - 1) / np.sqrt(dims)
    
    def analyze_distance_scaling(self, 
                               max_dim: int = 100,
                               n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Analyze how distances scale with dimension.
        
        Args:
            max_dim: Maximum dimension to analyze
            n_samples: Number of random samples per dimension
            
        Returns:
            Dictionary with analysis results
        """
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

class DistanceVisualizer:
    """Visualization tools for distance analysis."""
    
    def plot_signed_distances(self,
                            dims: np.ndarray,
                            delta_p0: np.ndarray,
                            delta_p1: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """Plot signed distances vs dimension.
        
        Args:
            dims: Array of dimensions
            delta_p0: Signed distances for zero vector
            delta_p1: Signed distances for ones vector
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(dims, delta_p0, 'bo-', label='p = (0, ..., 0)', linewidth=2)
        plt.plot(dims, delta_p1, 'ro-', label='p = (1, ..., 1)', linewidth=2)

        plt.xlabel('Dimension d', fontsize=12)
        plt.ylabel('Signed Distance Δd', fontsize=12)
        plt.title('Signed Distance vs Dimension', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.xticks(dims)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_distance_scaling(self,
                            analysis_results: Dict[str, np.ndarray],
                            save_path: Optional[str] = None) -> None:
        """Plot distance scaling analysis.
        
        Args:
            analysis_results: Results from distance scaling analysis
            save_path: Optional path to save plot
        """
        dims = analysis_results['dimensions']
        means = analysis_results['mean_distances']
        stds = analysis_results['std_distances']
        
        plt.figure(figsize=(10, 6))
        plt.plot(dims, means, 'b-', label='Mean Distance', linewidth=2)
        plt.fill_between(dims, means - stds, means + stds, alpha=0.2)
        
        plt.xlabel('Dimension', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.title('Distance Scaling with Dimension', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()