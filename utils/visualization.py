"""Visualization utilities for machine learning analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Union, Any, Tuple
from pathlib import Path
from core.logging import get_logger
from core.metrics import roc_curve, precision_recall_curve

# Configure logging using core logger
logger = get_logger(__name__)

def plot_learning_curves(history: Dict[str, List[float]],
                        title: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> None:
    """Plot training history curves.
    
    Args:
        history: Dictionary mapping metric names to lists of values
        title: Plot title (optional)
        figsize: Figure dimensions as (width, height) tuple
        save_path: Path to save figure (optional)
        
    Raises:
        ValueError: If history is empty or contains invalid data
    """
    if not history:
        raise ValueError("History dictionary cannot be empty")
        
    try:
        plt.figure(figsize=figsize)
        for metric_name, values in history.items():
            if not values:
                logger.warning(f"Empty values for metric {metric_name}")
                continue
            plt.plot(values, label=metric_name, linewidth=2)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Value', fontsize=12) 
        plt.title(title or 'Learning Curves', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting learning curves: {str(e)}")
        raise

def plot_decision_boundary(model: Any,
                         X: np.ndarray,
                         y: np.ndarray,
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> None:
    """Plot model decision boundary.
    
    Args:
        model: Fitted model with predict method
        X: Input features
        y: Target labels
        title: Plot title (optional)
        figsize: Figure dimensions
        save_path: Path to save figure (optional)
        
    Raises:
        ValueError: If inputs have invalid shape/dimensions
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plotting requires 2D input")
        
    try:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=figsize)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title(title or 'Decision Boundary', fontsize=14)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
            
    except Exception as e:
        logger.error(f"Error plotting decision boundary: {str(e)}")
        raise

def plot_roc_curve(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 6),
                  save_path: Optional[str] = None) -> None:
    """Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        title: Plot title (optional)
        figsize: Figure dimensions
        save_path: Path to save figure (optional)
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, 'b-', label='ROC curve', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title or 'ROC Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")
        raise

def plot_precision_recall_curve(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              title: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None) -> None:
    """Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        title: Plot title (optional)
        figsize: Figure dimensions
        save_path: Path to save figure (optional)
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, 'b-', label='PR curve', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title or 'Precision-Recall Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting precision-recall curve: {str(e)}")
        raise

class DistanceVisualizer:
    """Visualization tools for distance analysis."""
    
    def plot_signed_distances(self,
                            dims: np.ndarray,
                            delta_p0: np.ndarray,
                            delta_p1: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """Plot signed distances vs dimension."""
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
        """Plot distance scaling analysis."""
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