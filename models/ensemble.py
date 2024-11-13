"""Ensemble learning implementations."""

import numpy as np
from typing import List, Optional
from sklearn.base import BaseEstimator

class EnsembleLearner:
    """Ensemble learning implementation."""
    
    def __init__(self, models: List[BaseEstimator], weights: Optional[np.ndarray] = None):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all models in ensemble."""
        for model in self.models:
            model.fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted prediction from all models."""
        predictions = np.array([model.predict_proba(X)[:, 1] for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights) 