"""Time series preprocessing utilities."""

import numpy as np
from typing import Optional, List, Union, Tuple
from core import BaseTransformer
from core.validation import check_array

class TimeSeriesScaler(BaseTransformer):
    """Scale time series data while preserving temporal dependencies."""
    
    def __init__(self, window_size: int = 10, scale_method: str = 'standard'):
        self.window_size = window_size
        self.scale_method = scale_method
        self.statistics_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TimeSeriesScaler':
        """Compute rolling statistics."""
        X = check_array(X)
        
        if self.scale_method == 'standard':
            rolling_mean = np.array([
                np.mean(X[max(0, i-self.window_size):i+1], axis=0)
                for i in range(len(X))
            ])
            rolling_std = np.array([
                np.std(X[max(0, i-self.window_size):i+1], axis=0)
                for i in range(len(X))
            ])
            self.statistics_ = (rolling_mean, rolling_std)
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale time series data."""
        X = check_array(X)
        rolling_mean, rolling_std = self.statistics_
        
        if self.scale_method == 'standard':
            X_scaled = (X - rolling_mean) / (rolling_std + 1e-8)
            
        return X_scaled

class LagFeatureGenerator(BaseTransformer):
    """Generate lagged features for time series data."""
    
    def __init__(self, lags: List[int]):
        self.lags = sorted(lags)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LagFeatureGenerator':
        """Fit transformer (no-op)."""
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Generate lagged features."""
        X = check_array(X)
        n_samples, n_features = X.shape
        n_lags = len(self.lags)
        
        # Initialize output array
        X_lagged = np.zeros((n_samples, n_features * (n_lags + 1)))
        
        # Copy original features
        X_lagged[:, :n_features] = X
        
        # Generate lags
        for i, lag in enumerate(self.lags):
            start_idx = (i + 1) * n_features
            end_idx = (i + 2) * n_features
            X_lagged[lag:, start_idx:end_idx] = X[:-lag]
            
        return X_lagged

class SeasonalDecomposer(BaseTransformer):
    """Decompose time series into trend, seasonal, and residual components."""
    
    def __init__(self, period: int, model: str = 'additive'):
        self.period = period
        self.model = model
        self.components_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'SeasonalDecomposer':
        """Decompose time series."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        X = check_array(X)
        components = []
        
        for col in range(X.shape[1]):
            decomposition = seasonal_decompose(
                X[:, col],
                period=self.period,
                model=self.model
            )
            components.append({
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            })
            
        self.components_ = components
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return decomposed components."""
        X = check_array(X)
        n_samples, n_features = X.shape
        
        # Stack all components
        result = np.zeros((n_samples, n_features * 3))
        
        for i, comp in enumerate(self.components_):
            result[:, i] = comp['trend']
            result[:, i + n_features] = comp['seasonal']
            result[:, i + 2 * n_features] = comp['residual']
            
        return result 