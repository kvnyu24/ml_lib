"""Base classes for classification models."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
from core import (
    Estimator,
    Metric,
    Accuracy,
    get_metric
)

class BaseClassifier(Estimator, ABC):
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
            'accuracy': get_metric('accuracy')(y, y_pred),
            'precision': get_metric('precision')(y, y_pred, average='weighted'),
            'recall': get_metric('recall')(y, y_pred, average='weighted'),
            'f1': get_metric('f1')(y, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = get_metric('roc_auc')(y, y_proba, multi_class='ovr')
            
        return metrics