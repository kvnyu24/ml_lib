"""Categorical feature encoding utilities."""

import numpy as np
from typing import Optional, Dict, List, Union
from core import BaseTransformer
from core.validation import check_array

class CategoricalEncoder(BaseTransformer):
    """Base class for categorical encoders."""
    
    def __init__(self):
        self.mapping_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CategoricalEncoder':
        """Fit encoder."""
        raise NotImplementedError
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform categories."""
        raise NotImplementedError

class FrequencyEncoder(CategoricalEncoder):
    """Encode categorical variables using value frequencies."""
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FrequencyEncoder':
        """Calculate value frequencies."""
        X = check_array(X, dtype=object)
        
        self.mapping_ = {}
        for col in range(X.shape[1]):
            values, counts = np.unique(X[:, col], return_counts=True)
            self.mapping_[col] = dict(zip(values, counts / len(X)))
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform categories to frequencies."""
        X = check_array(X, dtype=object)
        result = np.zeros_like(X, dtype=float)
        
        for col in range(X.shape[1]):
            for row in range(X.shape[0]):
                result[row, col] = self.mapping_[col].get(X[row, col], 0)
                
        return result

class WOEEncoder(CategoricalEncoder):
    """Weight of Evidence encoding."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WOEEncoder':
        """Calculate WOE values."""
        X = check_array(X, dtype=object)
        y = check_array(y, ensure_2d=False)
        
        if len(np.unique(y)) != 2:
            raise ValueError("WOE encoding requires binary target")
            
        self.mapping_ = {}
        for col in range(X.shape[1]):
            self.mapping_[col] = {}
            unique_values = np.unique(X[:, col])
            
            for val in unique_values:
                mask = X[:, col] == val
                pos_rate = np.mean(y[mask] == 1)
                neg_rate = np.mean(y[mask] == 0)
                
                # Add smoothing to avoid division by zero
                eps = 1e-10
                woe = np.log((pos_rate + eps) / (neg_rate + eps))
                self.mapping_[col][val] = woe
                
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform categories to WOE values."""
        X = check_array(X, dtype=object)
        result = np.zeros_like(X, dtype=float)
        
        for col in range(X.shape[1]):
            for row in range(X.shape[0]):
                result[row, col] = self.mapping_[col].get(X[row, col], 0)
                
        return result 