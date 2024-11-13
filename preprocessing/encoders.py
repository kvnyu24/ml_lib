"""Feature encoding utilities."""

import numpy as np
from typing import Optional, List, Dict
from core import BaseTransformer
from core.validation import check_array

class LabelEncoder(BaseTransformer):
    """Encode categorical labels with value between 0 and n_classes-1."""
    
    def __init__(self):
        self.classes_ = None
        self.mapping_ = None
        
    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """Fit label encoder."""
        y = check_array(y, ensure_2d=False)
        self.classes_ = np.unique(y)
        self.mapping_ = {val: idx for idx, val in enumerate(self.classes_)}
        return self
        
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels to normalized encoding."""
        y = check_array(y, ensure_2d=False)
        return np.array([self.mapping_[val] for val in y])

class OneHotEncoder(BaseTransformer):
    """Encode categorical features as a one-hot numeric array."""
    
    def __init__(self, sparse: bool = False, handle_unknown: str = 'error'):
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.categories_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OneHotEncoder':
        """Fit OneHot encoder."""
        X = check_array(X)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using one-hot encoding."""
        X = check_array(X)
        n_samples, n_features = X.shape
        
        if self.sparse:
            # Implement sparse matrix output
            pass
        
        # Dense output
        result = np.zeros((n_samples, sum(len(cats) for cats in self.categories_)))
        
        col_idx = 0
        for i, cats in enumerate(self.categories_):
            idx = np.searchsorted(cats, X[:, i])
            result[np.arange(n_samples), col_idx + idx] = 1
            col_idx += len(cats)
            
        return result 