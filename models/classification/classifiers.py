"""Implementation of various classifiers."""

import numpy as np
from typing import List, Optional, Dict, Union
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from .base import BaseClassifier
from core.optimizers import BaseOptimizer, AdamOptimizer

class SoftmaxClassifier(BaseClassifier):
    """Softmax classifier with regularization and custom optimizer."""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 optimizer: Optional[BaseOptimizer] = None):
        self.C = C
        self.max_iter = max_iter
        self.optimizer = optimizer or AdamOptimizer()
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, multi_class='multinomial'
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier with kernel methods."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0,
                 gamma: Union[str, float] = 'scale'):
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class EnsembleClassifier(BaseClassifier):
    """Ensemble of multiple classifiers with voting."""
    
    def __init__(self, models: Optional[List[BaseEstimator]] = None,
                 weights: Optional[List[float]] = None):
        self.models = models or [
            LogisticRegression(),
            SVC(probability=True),
            RandomForestClassifier(),
            GradientBoostingClassifier()
        ]
        self.weights = weights or [1.0] * len(self.models)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for model in self.models:
            model.fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.average(probas, axis=0, weights=self.weights)