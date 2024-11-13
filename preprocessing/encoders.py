"""Feature encoding utilities."""

import numpy as np
from typing import Optional, List, Dict
from core import (
    Transformer,
    check_array,
    ValidationError,
    get_logger
)

class LabelEncoder(Transformer):
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

class OneHotEncoder(Transformer):
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

class OrdinalEncoder(Transformer):
    """Encode categorical features as an integer array."""
    
    def __init__(self, categories: Optional[List[List]] = None, handle_unknown: str = 'error'):
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.mappings_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OrdinalEncoder':
        """Fit ordinal encoder."""
        X = check_array(X)
        
        if self.categories is None:
            self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        else:
            self.categories_ = self.categories
            
        self.mappings_ = [{val: idx for idx, val in enumerate(cats)} 
                         for cats in self.categories_]
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using ordinal encoding."""
        X = check_array(X)
        n_samples, n_features = X.shape
        
        X_out = np.zeros_like(X, dtype=np.int64)
        
        for i, mapping in enumerate(self.mappings_):
            if self.handle_unknown == 'error':
                X_out[:, i] = [mapping[val] for val in X[:, i]]
            else:  # 'ignore'
                X_out[:, i] = [mapping.get(val, -1) for val in X[:, i]]
                
        return X_out

class TargetEncoder(Transformer):
    """Encode categorical features using target values."""
    
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.encodings_ = None
        self.global_mean_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TargetEncoder':
        """Fit target encoder."""
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        
        self.global_mean_ = np.mean(y)
        self.encodings_ = []
        
        for i in range(X.shape[1]):
            col_stats = {}
            unique_vals = np.unique(X[:, i])
            
            for val in unique_vals:
                mask = X[:, i] == val
                n_samples = np.sum(mask)
                col_mean = np.mean(y[mask]) if n_samples > 0 else self.global_mean_
                
                # Apply smoothing
                smoothed_mean = (
                    (n_samples * col_mean + self.smoothing * self.global_mean_) /
                    (n_samples + self.smoothing)
                )
                col_stats[val] = smoothed_mean
                
            self.encodings_.append(col_stats)
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using target encoding."""
        X = check_array(X)
        X_out = np.zeros_like(X, dtype=np.float64)
        
        for i, encoding in enumerate(self.encodings_):
            X_out[:, i] = [encoding.get(val, self.global_mean_) for val in X[:, i]]
            
        return X_out