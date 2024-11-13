"""Active learning implementations."""

import numpy as np
from typing import Optional, List, Union
from core import (
    Estimator,
    model_selection,
    DEFAULT_RANDOM_STATE,
    ValidationError,
    check_array
)

class ActiveLearner:
    """Active learning implementation with support for multiple query strategies.
    
    Strategies:
        - uncertainty: Select instances with highest prediction uncertainty
        - random: Random sampling from pool
    """
    
    SUPPORTED_STRATEGIES = ['uncertainty', 'random']
    
    def __init__(self, 
                 model: Estimator,
                 query_strategy: str = 'uncertainty',
                 random_state: Optional[int] = None):
        if not isinstance(model, Estimator):
            raise ValidationError("Model must be an instance of Estimator")
            
        if query_strategy not in self.SUPPORTED_STRATEGIES:
            raise ValidationError(
                f"Query strategy must be one of {self.SUPPORTED_STRATEGIES}"
            )
            
        self.model = model
        self.query_strategy = query_strategy
        self.random_state = random_state or DEFAULT_RANDOM_STATE
        self.rng = np.random.RandomState(self.random_state)
        
    def query(self, X_pool: np.ndarray, n_instances: int = 1) -> np.ndarray:
        """Select most informative instances for labeling.
        
        Args:
            X_pool: Array of unlabeled instances to select from
            n_instances: Number of instances to select
            
        Returns:
            Indices of selected instances
            
        Raises:
            ValidationError: If inputs are invalid
        """
        X_pool = check_array(X_pool)
        
        if n_instances < 1:
            raise ValidationError("n_instances must be greater than 0")
            
        if n_instances > len(X_pool):
            raise ValidationError(
                f"n_instances ({n_instances}) cannot be larger than "
                f"pool size ({len(X_pool)})"
            )
            
        if self.query_strategy == 'uncertainty':
            try:
                probas = model_selection.predict_proba(self.model, X_pool)
                uncertainties = 1 - np.max(probas, axis=1)
                return np.argsort(uncertainties)[-n_instances:]
            except Exception as e:
                # Fallback to random if uncertainty sampling fails
                self.rng.seed(self.random_state) 
                return self.rng.choice(len(X_pool), n_instances, replace=False)
                
        # Random sampling
        self.rng.seed(self.random_state)
        return self.rng.choice(len(X_pool), n_instances, replace=False)