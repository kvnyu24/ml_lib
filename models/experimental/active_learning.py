"""Active learning implementations."""

import numpy as np
from core.base import BaseEstimator

class ActiveLearner:
    """Active learning implementation."""
    
    def __init__(self, model: BaseEstimator, query_strategy: str = 'uncertainty'):
        self.model = model
        self.query_strategy = query_strategy
        
    def query(self, X_pool: np.ndarray, n_instances: int = 1) -> np.ndarray:
        """Select most informative instances for labeling."""
        if self.query_strategy == 'uncertainty':
            probas = self.model.predict_proba(X_pool)
            uncertainties = 1 - np.max(probas, axis=1)
            return np.argsort(uncertainties)[-n_instances:]
        return np.random.choice(len(X_pool), n_instances, replace=False) 