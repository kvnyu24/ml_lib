"""Data scaling and normalization utilities."""

import numpy as np
from typing import Optional, Union, Dict
from core import BaseTransformer
from core.validation import check_array

class StandardScaler(BaseTransformer):
    """Standardize features by removing mean and scaling to unit variance."""
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StandardScaler':
        """Compute mean and std to be used for scaling."""
        X = check_array(X)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization."""
        X = check_array(X)
        
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
            
        return X

class MinMaxScaler(BaseTransformer):
    """Scale features to a given range."""
    
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MinMaxScaler':
        """Compute min and max to be used for scaling."""
        X = check_array(X)
        
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        
        self.min_ = data_min
        self.scale_ = (data_max - data_min)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform scaling to the [0, 1] range."""
        X = check_array(X)
        
        X_std = (X - self.min_) / self.scale_
        X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        return X_scaled 