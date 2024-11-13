"""Automated machine learning implementation."""

import numpy as np
import time
from typing import List, Dict
from core.base import BaseEstimator
from ..evaluation import KFoldCV, get_metric

class AutoML:
    """Automated machine learning implementation."""
    
    def __init__(self, models: List[BaseEstimator], 
                 param_distributions: List[Dict],
                 max_time: int = 3600,
                 cv: int = 5,
                 metric: str = 'accuracy'):
        self.models = models
        self.param_distributions = param_distributions
        self.max_time = max_time
        self.best_model = None
        self.cv = KFoldCV(n_splits=cv)
        self.metric = get_metric(metric)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Automatically select and tune best model."""
        best_score = float('-inf')
        start_time = time.time()
        
        for model, params in zip(self.models, self.param_distributions):
            if time.time() - start_time > self.max_time:
                break
                
            scores = []
            for train_idx, val_idx in self.cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.set_params(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(self.metric(y_val, y_pred))
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                self.best_model = model