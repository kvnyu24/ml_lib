"""Missing value imputation utilities."""

import numpy as np
from typing import Optional, Union, Dict
from scipy.stats import mode
from models.neighbors import KNeighborsRegressor
from core import (
    Transformer, 
    check_array,
    ValidationError,
    get_logger
)

logger = get_logger(__name__)

class MissingValueImputer(Transformer):
    """Imputation transformer for completing missing values."""
    
    def __init__(self, strategy: str = 'mean', fill_value: Optional[Union[str, float]] = None):
        """
        Args:
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            fill_value: Value to use for constant imputation
        """
        valid_strategies = ['mean', 'median', 'most_frequent', 'constant']
        if strategy not in valid_strategies:
            raise ValidationError(f"strategy must be one of {valid_strategies}")
            
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MissingValueImputer':
        """Compute the imputation statistics."""
        X = check_array(X)
        
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = np.array([
                mode(column[~np.isnan(column)]).mode[0]
                for column in X.T
            ])
        elif self.strategy == 'constant':
            if self.fill_value is None:
                raise ValidationError("fill_value must be provided for constant strategy")
            self.statistics_ = np.full(X.shape[1], self.fill_value)
            
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values."""
        X = check_array(X)
        
        if self.statistics_ is None:
            raise ValidationError("MissingValueImputer must be fitted before transform")
            
        if np.any(np.isnan(X)):
            mask = np.isnan(X)
            X = X.copy()
            X[mask] = self.statistics_[np.where(mask)[1]]
            
        return X

class KNNImputer(Transformer):
    """Imputation using k-Nearest Neighbors."""
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        """
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function used ('uniform' or 'distance')
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValidationError("n_neighbors must be a positive integer")
            
        if weights not in ['uniform', 'distance']:
            raise ValidationError("weights must be 'uniform' or 'distance'")
            
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_ = None
        self.train_data_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KNNImputer':
        """Fit the imputer."""
        X = check_array(X)
        self.train_data_ = X.copy()
        
        # Initialize kNN model
        self.knn_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        
        # Fit on non-missing data
        mask = ~np.isnan(X).any(axis=1)
        if not np.any(mask):
            raise ValidationError("No complete samples available for fitting")
        self.knn_.fit(X[mask], X[mask])
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values using kNN."""
        X = check_array(X)
        X_imputed = X.copy()
        
        if self.knn_ is None:
            raise ValidationError("KNNImputer must be fitted before transform")
            
        # Find samples with missing values
        missing_rows = np.where(np.isnan(X).any(axis=1))[0]
        
        for row in missing_rows:
            missing_cols = np.where(np.isnan(X[row]))[0]
            
            # Find k nearest neighbors with non-missing values
            valid_mask = ~np.isnan(X[row])
            if not np.any(valid_mask):
                continue
                
            predictions = self.knn_.predict(
                X[row, valid_mask].reshape(1, -1)
            )
            
            X_imputed[row, missing_cols] = predictions[0][missing_cols]
            
        return X_imputed

class TimeSeriesImputer(Transformer):
    """Imputation for time series data."""
    
    def __init__(self, method: str = 'linear', max_gap: Optional[int] = None):
        """
        Args:
            method: Interpolation method ('linear', 'forward', 'backward')
            max_gap: Maximum gap size to interpolate
        """
        valid_methods = ['linear', 'forward', 'backward']
        if method not in valid_methods:
            raise ValidationError(f"method must be one of {valid_methods}")
            
        self.method = method
        self.max_gap = max_gap
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TimeSeriesImputer':
        """Fit the imputer."""
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values in time series."""
        X = check_array(X)
        X_imputed = X.copy()
        
        for col in range(X.shape[1]):
            missing_mask = np.isnan(X[:, col])
            if not np.any(missing_mask):
                continue
                
            valid_indices = np.where(~missing_mask)[0]
            missing_indices = np.where(missing_mask)[0]
            
            if len(valid_indices) == 0:
                logger.warning(f"Column {col} has all missing values")
                continue
                
            if self.method == 'linear':
                # Linear interpolation
                X_imputed[missing_indices, col] = np.interp(
                    missing_indices,
                    valid_indices,
                    X[valid_indices, col]
                )
            elif self.method == 'forward':
                # Forward fill
                last_valid = float('nan')
                for i in range(len(X)):
                    if not np.isnan(X[i, col]):
                        last_valid = X[i, col]
                    elif not np.isnan(last_valid):
                        if self.max_gap is None or i - valid_indices[valid_indices < i][-1] <= self.max_gap:
                            X_imputed[i, col] = last_valid
            else:  # backward
                # Backward fill
                next_valid = float('nan')
                for i in range(len(X)-1, -1, -1):
                    if not np.isnan(X[i, col]):
                        next_valid = X[i, col]
                    elif not np.isnan(next_valid):
                        if self.max_gap is None or valid_indices[valid_indices > i][0] - i <= self.max_gap:
                            X_imputed[i, col] = next_valid
                            
        return X_imputed
