"""Data scaling and normalization utilities."""

import numpy as np
from scipy.special import erfinv
from typing import Optional, Union, Dict
from core import Transformer, check_array

class StandardScaler(Transformer):
    """Standardize features by removing mean and scaling to unit variance."""
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StandardScaler':
        """Compute mean and std to be used for scaling."""
        X = check_array(X)
        
        # Validate input array is not empty
        if X.shape[0] == 0:
            raise ValueError("Cannot fit StandardScaler with empty array")
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize features."""
        X = check_array(X)
        X_transformed = X.copy()
        
        if self.with_mean and self.mean_ is not None:
            X_transformed -= self.mean_
        if self.with_std and self.scale_ is not None:
            X_transformed /= self.scale_
            
        return X_transformed

class MinMaxScaler(Transformer):
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

class RobustScaler(Transformer):
    """Scale features using statistics that are robust to outliers."""
    
    def __init__(self, with_centering: bool = True, with_scaling: bool = True, 
                 quantile_range: tuple = (25.0, 75.0)):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'RobustScaler':
        """Compute median and quantiles to be used for scaling."""
        X = check_array(X)
        
        q_min, q_max = self.quantile_range
        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        if self.with_scaling:
            q = np.percentile(X, [q_min, q_max], axis=0)
            self.scale_ = (q[1] - q[0])
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform robust scaling."""
        X = check_array(X)
        
        if self.with_centering:
            X = X - self.center_
        if self.with_scaling:
            X = X / self.scale_
            
        return X

class QuantileScaler(Transformer):
    """Transform features using quantile information."""
    
    def __init__(self, n_quantiles: int = 1000, output_distribution: str = 'uniform'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.quantiles_ = None
        self.references_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'QuantileScaler':
        """Compute quantiles for the transformation."""
        X = check_array(X)
        
        self.quantiles_ = []
        for feature_idx in range(X.shape[1]):
            quantiles = np.percentile(
                X[:, feature_idx],
                np.linspace(0, 100, self.n_quantiles)
            )
            self.quantiles_.append(quantiles)
            
        self.references_ = np.linspace(0, 1, self.n_quantiles)
        if self.output_distribution == 'normal':
            self.references_ = np.sqrt(2) * erfinv(2 * self.references_ - 1)
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using quantile transformation."""
        X = check_array(X)

        if self.quantiles_ is None or self.references_ is None:
            raise ValueError("QuantileScaler must be fitted before calling transform")

        X_transformed = np.zeros_like(X)
        
        for feature_idx in range(X.shape[1]):
            X_transformed[:, feature_idx] = np.interp(
                X[:, feature_idx].ravel(),  # Input x-coordinates
                np.array(self.quantiles_[feature_idx]).ravel(),  # Reference x-coordinates
                np.array(self.references_).ravel()  # Reference y-coordinates
            )
            
        return X_transformed
