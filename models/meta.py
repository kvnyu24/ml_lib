"""Meta-learning implementations."""

import numpy as np
from typing import List
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

class MetaLearner:
    """Meta-learning implementation."""
    
    def __init__(self, base_models: List[BaseEstimator]):
        self.base_models = base_models
        self.meta_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train meta-learner on base model predictions."""
        base_predictions = np.column_stack([
            model.fit(X, y).predict_proba(X)[:, 1] 
            for model in self.base_models
        ])
        self.meta_model = LogisticRegression().fit(base_predictions, y) 