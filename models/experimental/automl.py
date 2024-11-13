"""Automated machine learning implementation."""

import numpy as np
import time
from typing import List, Dict, Optional, Union
from core import (
    Estimator,
    get_metric,
    check_array,
    check_X_y,
    ValidationError
)
from models.evaluation import KFoldCV

class AutoML:
    """Automated machine learning implementation with robust model selection and tuning."""
    
    def __init__(self, 
                 models: List[Estimator], 
                 param_distributions: List[Dict],
                 max_time: int = 3600,
                 cv: int = 5,
                 metric: str = 'accuracy',
                 n_trials: int = 10,
                 random_state: Optional[int] = None):
        """Initialize AutoML.
        
        Args:
            models: List of model instances to try
            param_distributions: List of parameter distributions for each model
            max_time: Maximum time in seconds for model search
            cv: Number of cross-validation folds
            metric: Metric to optimize ('accuracy', 'f1', etc)
            n_trials: Number of random parameter combinations to try per model
            random_state: Random seed for reproducibility
        """
        if len(models) != len(param_distributions):
            raise ValidationError("Number of models must match number of parameter distributions")
            
        self.models = models
        self.param_distributions = param_distributions
        self.max_time = max_time
        self.cv = KFoldCV(n_splits=cv, shuffle=True, random_state=random_state)
        self.metric = get_metric(metric)
        self.n_trials = n_trials
        self.random_state = random_state
        
        self.best_model = None
        self.best_params = None
        self.best_score = float('-inf')
        self.model_scores = {}
        
        np.random.seed(random_state)
        
    def _sample_params(self, param_dist: Dict) -> Dict:
        """Sample parameters from distributions."""
        params = {}
        for param, dist in param_dist.items():
            if isinstance(dist, (list, tuple)):
                params[param] = np.random.choice(dist)
            elif isinstance(dist, dict):
                if dist.get('type') == 'int':
                    params[param] = np.random.randint(dist['min'], dist['max'])
                elif dist.get('type') == 'float':
                    params[param] = np.random.uniform(dist['min'], dist['max'])
        return params
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AutoML':
        """Automatically select and tune best model."""
        X, y = check_X_y(X, y)
        start_time = time.time()
        
        for model_idx, (model, param_dist) in enumerate(zip(self.models, self.param_distributions)):
            model_scores = []
            
            for trial in range(self.n_trials):
                if time.time() - start_time > self.max_time:
                    break
                    
                # Sample random parameters
                params = self._sample_params(param_dist)
                
                try:
                    scores = []
                    for train_idx, val_idx in self.cv.split(X, y):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        model_copy = model.__class__(**params)
                        model_copy.fit(X_train, y_train)
                        y_pred = model_copy.predict(X_val)
                        scores.append(self.metric(y_val, y_pred))
                    
                    avg_score = np.mean(scores)
                    model_scores.append((avg_score, params))
                    
                    if avg_score > self.best_score:
                        self.best_score = avg_score
                        self.best_params = params
                        self.best_model = model.__class__(**params)
                        self.best_model.fit(X, y)
                        
                except Exception as e:
                    print(f"Error fitting model {model.__class__.__name__} with params {params}: {str(e)}")
                    continue
            
            if model_scores:
                self.model_scores[model.__class__.__name__] = model_scores
                
        if self.best_model is None:
            raise ValidationError("No models were successfully trained")
            
        return self