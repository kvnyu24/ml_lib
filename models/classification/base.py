"""Base classes for classification models."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
from core.metrics import accuracy, precision, recall, f1, roc_auc
from core.base import BaseEstimator

class BaseClassifier(BaseEstimator, ABC):
    """Abstract base class for classifiers."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier."""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate classifier performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X) if hasattr(self, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy(y, y_pred),
            'precision': precision(y, y_pred, average='weighted'),
            'recall': recall(y, y_pred, average='weighted'),
            'f1': f1(y, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc(y, y_proba, multi_class='ovr')
            
        return metrics