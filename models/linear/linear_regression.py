"""
Linear Regression and Optimization Library
========================================

A comprehensive library for linear regression and optimization:

- Multiple regression algorithms (OLS, Ridge, Lasso, Elastic Net)
- Advanced optimization methods (SGD, Adam, RMSprop, Momentum)
- Regularization techniques
- Feature engineering and selection
- Cross-validation and model selection
- Model evaluation and metrics
- Visualization utilities
- Extensible interfaces for custom models

The implementation follows clean design principles with modular components.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core import (
    Estimator, Loss,
    check_array, check_X_y, check_is_fitted,
    get_logger, TrainingLogger,
    Number, Array, Features, Target,
    EPSILON, DEFAULT_RANDOM_STATE
)

from optimization.optimizers import AdamOptimizer, RMSpropOptimizer
from evaluation import ModelEvaluator

# Configure logging
logger = get_logger(__name__)

class ElasticNetLoss(Loss):
    """Elastic Net loss function with L1 and L2 regularization."""
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(y_pred))
        l2_penalty = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(y_pred**2)
        return mse + l1_penalty + l2_penalty
        
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        grad_mse = -2 * (y_true - y_pred)
        grad_l1 = self.alpha * self.l1_ratio * np.sign(y_pred)
        grad_l2 = self.alpha * (1 - self.l1_ratio) * y_pred
        return grad_mse + grad_l1 + grad_l2

class ElasticNetRegression(Estimator):
    """Elastic Net regression with L1 and L2 regularization."""
    
    def __init__(self,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 optimizer: Optional[AdamOptimizer] = None,
                 fit_intercept: bool = True):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.optimizer = optimizer or AdamOptimizer()
        self.fit_intercept = fit_intercept
        self.loss = ElasticNetLoss(alpha=alpha, l1_ratio=l1_ratio)
        
    def fit(self, X: Features, y: Target) -> 'ElasticNetRegression':
        X, y = check_X_y(X, y)
        
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        self._input_dim = X.shape[1]
        self.weights = np.zeros(self._input_dim)
        
        for _ in range(1000):  # Max iterations
            y_pred = self.predict(X)
            gradients = self.loss.gradient(y, y_pred)
            self.weights = self.optimizer.compute_update(self.weights, gradients)
            
        self._is_fitted = True
        return self
        
    def predict(self, X: Features) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        return X @ self.weights

class Visualizer:
    """Visualization utilities."""
    
    @staticmethod
    def plot_regression(X: Features,
                       y: Target,
                       y_pred: np.ndarray,
                       title: str = 'Regression Plot') -> None:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Prediction')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def plot_learning_curve(history: Dict[str, List],
                          title: str = 'Learning Curve') -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        plt.show()