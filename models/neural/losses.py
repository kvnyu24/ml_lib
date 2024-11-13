"""Neural network loss functions."""

import numpy as np
from typing import Optional
from core import Loss, EPSILON

class MSELoss(Loss):
    """Mean squared error loss."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
        
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * (y_true - y_pred) / y_true.shape[0]

class CrossEntropyLoss(Loss):
    """Cross entropy loss for classification."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        if y_true.ndim == 1:
            return -np.mean(y_true * np.log(y_pred) + 
                          (1 - y_true) * np.log(1 - y_pred))
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        if y_true.ndim == 1:
            return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
        return -(y_true / y_pred) / y_true.shape[0] 