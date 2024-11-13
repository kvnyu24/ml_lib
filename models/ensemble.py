"""Ensemble learning implementations."""

import numpy as np
from typing import List, Optional, Union, Callable
from core import Estimator, Loss

class GradientBoostingLoss(Loss):
    """Loss function for gradient boosting."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
        
    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_true - y_pred

class BaseBooster(Estimator):
    """Base class for gradient boosting models."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 subsample: float = 1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.estimators_ = []
        self.loss = GradientBoostingLoss()
        
    def _boost(self, X: np.ndarray, y: np.ndarray, residuals: np.ndarray):
        raise NotImplementedError
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimators_ = []
        F = np.zeros(len(y))
        
        for _ in range(self.n_estimators):
            residuals = self.loss.negative_gradient(y, F)
            estimator = self._boost(X, y, residuals)
            self.estimators_.append(estimator)
            F += self.learning_rate * estimator.predict(X)
            
        return self

class GradientBoostingRegressor(BaseBooster):
    """Custom gradient boosting regressor implementation."""
    
    def _boost(self, X: np.ndarray, y: np.ndarray, residuals: np.ndarray):
        subsample_mask = np.random.choice(len(y), 
                                        size=int(len(y) * self.subsample),
                                        replace=False)
        tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                   min_samples_split=self.min_samples_split)
        tree.fit(X[subsample_mask], residuals[subsample_mask])
        return tree
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.zeros(len(X))
        for estimator in self.estimators_:
            pred += self.learning_rate * estimator.predict(X)
        return pred

class XGBoostRegressor(BaseBooster):
    """XGBoost-style implementation with custom split finding."""
    
    def _find_split(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> tuple:
        """Find best split using second-order approximation."""
        best_gain = 0
        best_split = None
        
        for feature in range(X.shape[1]):
            G_left = 0
            H_left = 0
            G_right = g.sum()
            H_right = h.sum()
            
            sorted_idx = np.argsort(X[:, feature])
            
            for i in range(len(sorted_idx) - 1):
                idx = sorted_idx[i]
                G_left += g[idx]
                H_left += h[idx]
                G_right -= g[idx]
                H_right -= h[idx]
                
                if X[sorted_idx[i], feature] == X[sorted_idx[i + 1], feature]:
                    continue
                    
                gain = (G_left**2 / (H_left + 1) + 
                       G_right**2 / (H_right + 1) - 
                       (G_left + G_right)**2 / (H_left + H_right + 1))
                       
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, X[sorted_idx[i], feature])
                    
        return best_split
        
    def _boost(self, X: np.ndarray, y: np.ndarray, residuals: np.ndarray):
        g = residuals  # First order gradient
        h = np.ones_like(residuals)  # Second order gradient
        tree = self._build_tree(X, g, h)
        return tree

class LightGBMRegressor(BaseBooster):
    """LightGBM-style implementation with histogram-based learning."""
    
    def __init__(self, *args, n_bins: int = 255, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins
        
    def _create_hist(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> List[np.ndarray]:
        """Create histogram bins for features."""
        histograms = []
        for feature in range(X.shape[1]):
            bins = np.linspace(X[:, feature].min(), X[:, feature].max(), self.n_bins)
            indices = np.digitize(X[:, feature], bins)
            hist_g = np.bincount(indices, weights=g, minlength=self.n_bins+1)
            hist_h = np.bincount(indices, weights=h, minlength=self.n_bins+1)
            histograms.append((hist_g, hist_h))
        return histograms
        
    def _boost(self, X: np.ndarray, y: np.ndarray, residuals: np.ndarray):
        g = residuals
        h = np.ones_like(residuals)
        histograms = self._create_hist(X, g, h)
        tree = self._build_hist_tree(X, histograms)
        return tree