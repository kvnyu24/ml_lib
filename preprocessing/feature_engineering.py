"""Feature engineering utilities."""

import numpy as np
from typing import Optional, List, Union
from itertools import combinations
from core import (
    Transformer,
    check_array,
    get_logger
)

logger = get_logger(__name__)

class PolynomialFeatures(Transformer):
    """Generate polynomial and interaction features."""
    
    def __init__(self, degree: int = 2, interaction_only: bool = False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.n_output_features_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PolynomialFeatures':
        """Compute number of output features."""
        X = check_array(X)
        n_features = X.shape[1]
        
        combinations_list = []
        for d in range(0, self.degree + 1):
            if self.interaction_only and d > 1:
                combinations_list.extend(combinations(range(n_features), d))
            else:
                combinations_list.extend(combinations_with_replacement(range(n_features), d))
                
        self.n_output_features_ = len(combinations_list)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to polynomial features."""
        X = check_array(X)
        n_samples, n_features = X.shape
        
        # Implementation details for polynomial feature generation
        # This is a simplified version - full implementation would be more complex
        if self.degree == 2:
            X_new = np.column_stack([
                X,
                np.multiply.outer(X, X).reshape(n_samples, -1)
            ])
            
        return X_new 