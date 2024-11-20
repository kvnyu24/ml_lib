"""Ensemble learning implementations."""

import numpy as np
from typing import List, Optional, Union, Callable
from core import Estimator, Loss
from models.trees.decision_tree import DecisionTreeRegressor, Node

class GradientBoostingLoss(Loss):
    """Loss function for gradient boosting."""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))
        
    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_true - y_pred
        
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient."""
        return y_pred - y_true

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
    
    def _find_split(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> Optional[tuple]:
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
        
    def _build_tree(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int = 0) -> Node:
        """Build tree recursively using gradient statistics."""
        if depth >= self.max_depth:
            return Node(value=np.array([-g.sum() / (h.sum() + 1)]), is_leaf=True)
            
        split = self._find_split(X, g, h)
        if split is None:
            return Node(value=np.array([-g.sum() / (h.sum() + 1)]), is_leaf=True)
            
        feature_idx, threshold = split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        
        return Node(feature_idx=feature_idx, threshold=threshold, 
                   left=left_child, right=right_child)
        
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
        
    def _build_hist_tree(self, X: np.ndarray, histograms: List[np.ndarray], depth: int = 0) -> Node:
        """Build tree using histogram-based splits."""
        if depth >= self.max_depth:
            return Node(value=np.array([0.0]), is_leaf=True)
            
        best_gain = 0
        best_split = None
        
        for feature, (hist_g, hist_h) in enumerate(histograms):
            G_left = 0
            H_left = 0
            G_right = hist_g.sum()
            H_right = hist_h.sum()
            
            for bin_idx in range(len(hist_g) - 1):
                G_left += hist_g[bin_idx]
                H_left += hist_h[bin_idx]
                G_right -= hist_g[bin_idx]
                H_right -= hist_h[bin_idx]
                
                gain = (G_left**2 / (H_left + 1) + 
                       G_right**2 / (H_right + 1) - 
                       (G_left + G_right)**2 / (H_left + H_right + 1))
                       
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, bin_idx)
                    
        if best_split is None:
            return Node(value=np.array([0.0]), is_leaf=True)
            
        feature_idx, bin_idx = best_split
        threshold = np.linspace(X[:, feature_idx].min(), 
                              X[:, feature_idx].max(), 
                              self.n_bins)[bin_idx]
                              
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_histograms = []
        right_histograms = []
        for feature in range(X.shape[1]):
            hist_g, hist_h = histograms[feature]
            left_hist_g = np.bincount(np.digitize(X[left_mask, feature], 
                                    np.linspace(X[:, feature].min(), X[:, feature].max(), self.n_bins)),
                                    minlength=self.n_bins+1)
            left_hist_h = np.bincount(np.digitize(X[left_mask, feature],
                                    np.linspace(X[:, feature].min(), X[:, feature].max(), self.n_bins)),
                                    minlength=self.n_bins+1)
            right_hist_g = hist_g - left_hist_g
            right_hist_h = hist_h - left_hist_h
            left_histograms.append((left_hist_g, left_hist_h))
            right_histograms.append((right_hist_g, right_hist_h))
            
        left_child = self._build_hist_tree(X[left_mask], left_histograms, depth + 1)
        right_child = self._build_hist_tree(X[right_mask], right_histograms, depth + 1)
        
        return Node(feature_idx=feature_idx, threshold=threshold,
                   left=left_child, right=right_child)
        
    def _boost(self, X: np.ndarray, y: np.ndarray, residuals: np.ndarray):
        g = residuals
        h = np.ones_like(residuals)
        histograms = self._create_hist(X, g, h)
        tree = self._build_hist_tree(X, histograms)
        return tree