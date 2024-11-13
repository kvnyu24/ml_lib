"""Base classes for classification models."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class BaseClassifier(ABC):
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
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_proba, multi_class='ovr')
            
        return metrics 