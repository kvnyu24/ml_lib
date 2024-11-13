"""Active learning strategies and algorithms."""

import numpy as np
from typing import Optional, Callable, List
from core.base import BaseEstimator
from models.classification import BaseClassifier

class ActiveLearner:
    """Active learning wrapper for any classifier."""
    
    def __init__(self, 
                 estimator: BaseClassifier,
                 query_strategy: str = 'uncertainty',
                 batch_size: int = 1):
        self.estimator = estimator
        self.query_strategy = query_strategy
        self.batch_size = batch_size
        self.labeled_indices: List[int] = []
        
    def query(self, X_pool: np.ndarray) -> List[int]:
        """Select most informative instances for labeling."""
        if self.query_strategy == 'uncertainty':
            probas = self.estimator.predict_proba(X_pool)
            uncertainties = 1 - np.max(probas, axis=1)
            return np.argsort(uncertainties)[-self.batch_size:]
        elif self.query_strategy == 'diversity':
            # Implement diversity-based sampling
            pass 