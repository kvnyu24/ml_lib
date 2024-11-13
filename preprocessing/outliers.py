"""Outlier detection utilities."""

import numpy as np
from typing import Optional, List
from core import BaseTransformer
from core.validation import check_array

class OutlierDetector(BaseTransformer):
    """Base class for outlier detection."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OutlierDetector':
        """Fit the outlier detector."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if instances are outliers."""
        raise NotImplementedError

class IQROutlierDetector(OutlierDetector):
    """Outlier detection using Interquartile Range."""
    
    def __init__(self, contamination: float = 0.1):
        super().__init__(contamination)
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'IQROutlierDetector':
        """Compute IQR statistics."""
        X = check_array(X)
        
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outliers using IQR method."""
        X = check_array(X)
        
        lower_bound = self.q1_ - 1.5 * self.iqr_
        upper_bound = self.q3_ + 1.5 * self.iqr_
        
        mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        return mask 