"""Outlier detection utilities."""

import numpy as np
from typing import Optional, List
from core import (
    Transformer,
    check_array
)

class OutlierDetector(Transformer):
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

class IsolationForest(OutlierDetector):
    """Isolation Forest for outlier detection."""
    
    def __init__(self, contamination: float = 0.1, n_trees: int = 100, max_samples: int = 256):
        super().__init__(contamination)
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.trees_ = []
        self.threshold_ = None
        
    def _build_tree(self, X: np.ndarray, height_limit: int) -> dict:
        """Build a single isolation tree."""
        n_samples, n_features = X.shape
        
        if height_limit == 0 or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}
            
        feature = np.random.randint(n_features)
        split_value = np.random.uniform(
            X[:, feature].min(),
            X[:, feature].max()
        )
        
        left_mask = X[:, feature] < split_value
        right_mask = ~left_mask
        
        return {
            'type': 'split',
            'feature': feature,
            'value': split_value,
            'left': self._build_tree(X[left_mask], height_limit - 1),
            'right': self._build_tree(X[right_mask], height_limit - 1)
        }
        
    def _path_length(self, x: np.ndarray, tree: dict, current_height: int = 0) -> float:
        """Compute path length for a sample."""
        if tree['type'] == 'leaf':
            return current_height
            
        if x[tree['feature']] < tree['value']:
            return self._path_length(x, tree['left'], current_height + 1)
        return self._path_length(x, tree['right'], current_height + 1)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'IsolationForest':
        """Build isolation forest."""
        X = check_array(X)
        n_samples = min(self.max_samples, X.shape[0])
        height_limit = int(np.ceil(np.log2(n_samples)))
        
        self.trees_ = []
        for _ in range(self.n_trees):
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            tree = self._build_tree(X[indices], height_limit)
            self.trees_.append(tree)
            
        # Compute threshold
        scores = -self.score_samples(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        
        return self
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        X = check_array(X)
        scores = np.zeros(X.shape[0])
        
        for x_idx, x in enumerate(X):
            paths = [self._path_length(x, tree) for tree in self.trees_]
            avg_path = np.mean(paths)
            scores[x_idx] = -avg_path
            
        return scores
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if instances are outliers."""
        scores = self.score_samples(X)
        return scores >= self.threshold_

class LocalOutlierFactor(OutlierDetector):
    """Local Outlier Factor for outlier detection."""
    
    def __init__(self, contamination: float = 0.1, n_neighbors: int = 20):
        super().__init__(contamination)
        self.n_neighbors = n_neighbors
        self.X_ = None
        self.threshold_ = None
        self.lrd_ = None
        
    def _distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))
        
    def _k_distance(self, distances: np.ndarray) -> np.ndarray:
        """Compute k-distance for each point."""
        sorted_distances = np.sort(distances, axis=1)
        return sorted_distances[:, self.n_neighbors - 1]
        
    def _reachability_distance(self, distances: np.ndarray, k_distances: np.ndarray) -> np.ndarray:
        """Compute reachability distance matrix."""
        reach_dist = np.maximum(distances, k_distances[np.newaxis, :])
        return reach_dist
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LocalOutlierFactor':
        """Fit LOF detector."""
        X = check_array(X)
        self.X_ = X
        
        # Compute distances and k-distances
        distances = self._distances(X, X)
        k_distances = self._k_distance(distances)
        reach_dist = self._reachability_distance(distances, k_distances)
        
        # Compute local reachability density
        k_neighbors = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        self.lrd_ = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            neighbors = k_neighbors[i]
            self.lrd_[i] = self.n_neighbors / np.sum(reach_dist[i, neighbors])
            
        # Compute LOF scores
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            neighbors = k_neighbors[i]
            scores[i] = np.mean(self.lrd_[neighbors]) / self.lrd_[i]
            
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if instances are outliers."""
        X = check_array(X)
        
        # Compute distances to training points
        distances = self._distances(X, self.X_)
        k_distances = self._k_distance(distances)
        reach_dist = self._reachability_distance(distances, k_distances)
        
        # Compute LOF scores
        k_neighbors = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        scores = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            neighbors = k_neighbors[i]
            lrd = self.n_neighbors / np.sum(reach_dist[i, neighbors])
            scores[i] = np.mean(self.lrd_[neighbors]) / lrd
            
        return scores > self.threshold_