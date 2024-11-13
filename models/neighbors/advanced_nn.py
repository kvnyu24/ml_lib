"""K-Nearest Neighbors advanced implementations."""

from typing import Literal, Optional, Tuple
import numpy as np
from scipy.spatial import Voronoi

from core.base import Estimator
from core.validation import check_array, check_X_y, check_is_fitted
from core.exceptions import ValidationError

class AdaptiveKNN(Estimator):
    """KNN classifier with adaptive neighborhood size."""
    
    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 50,
        weights: Literal['uniform', 'distance'] = 'distance'
    ):
        self.min_k = min_k
        self.max_k = max_k
        self.weights = weights
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveKNN':
        """Fit the model using X as training data and y as target values."""
        # Validate inputs
        X, y = check_X_y(X, y)
        
        if self.min_k < 1:
            raise ValidationError("min_k must be greater than zero")
            
        if self.max_k > len(X):
            raise ValidationError(
                f"max_k ({self.max_k}) cannot be larger than "
                f"number of samples ({len(X)})"
            )
            
        self.X_train = X
        self.y_train = y
        
        # Compute optimal k for each training point
        self.optimal_k = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            distances = np.linalg.norm(X - X[i], axis=1)
            sorted_idx = np.argsort(distances)[1:] # Exclude self
            
            # Find k that minimizes leave-one-out error
            min_error = float('inf')
            best_k = self.min_k
            
            for k in range(self.min_k, min(self.max_k, len(sorted_idx))):
                neighbors = sorted_idx[:k]
                if self.weights == 'distance':
                    weights = 1 / (distances[neighbors] + np.finfo(float).eps)
                    pred = np.bincount(y[neighbors], weights=weights)
                else:
                    pred = np.bincount(y[neighbors])
                pred = np.argmax(pred)
                
                error = pred != y[i]
                if error < min_error:
                    min_error = error
                    best_k = k
                    
            self.optimal_k[i] = best_k
            
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        check_is_fitted(self, ['X_train', 'y_train', 'optimal_k'])
        X = check_array(X)
        
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k = int(np.mean(self.optimal_k)) # Use average optimal k
            neighbors = np.argsort(distances)[:k]
            
            if self.weights == 'distance':
                weights = 1 / (distances[neighbors] + np.finfo(float).eps)
                pred = np.bincount(self.y_train[neighbors], weights=weights)
            else:
                pred = np.bincount(self.y_train[neighbors])
                
            predictions[i] = np.argmax(pred)
            
        return predictions

class LocallyWeightedRegressor(Estimator):
    """Non-parametric locally weighted regression."""
    
    def __init__(
        self,
        kernel: Literal['gaussian', 'epanechnikov'] = 'gaussian',
        bandwidth: float = 1.0
    ):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.kernels = {
            'gaussian': lambda d: np.exp(-d**2 / (2 * self.bandwidth**2)),
            'epanechnikov': lambda d: np.maximum(0, (1 - (d/self.bandwidth)**2))
        }
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LocallyWeightedRegressor':
        """Store the training data."""
        X, y = check_X_y(X, y)
        self.X_train = X
        self.y_train = y
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples in X."""
        check_is_fitted(self, ['X_train', 'y_train'])
        X = check_array(X)
        
        predictions = np.zeros(len(X))
        kernel_func = self.kernels[self.kernel]
        
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.X_train - x, axis=1)
            weights = kernel_func(distances)
            
            # Weighted least squares
            W = np.diag(weights)
            X_w = self.X_train.T @ W @ self.X_train
            y_w = self.X_train.T @ W @ self.y_train
            
            try:
                beta = np.linalg.solve(X_w, y_w)
                predictions[i] = x @ beta
            except:
                # Fallback to weighted average if matrix is singular
                predictions[i] = np.average(self.y_train, weights=weights)
                
        return predictions

class VoronoiClassifier(Estimator):
    """Classification based on Voronoi diagram of training points."""
    
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VoronoiClassifier':
        """Fit the Voronoi classifier."""
        X, y = check_X_y(X, y)
        self.X_train = X
        self.y_train = y
        
        # Compute Voronoi diagram
        self.vor = Voronoi(X)
        
        # Assign labels to Voronoi regions
        self.region_labels = {}
        for i, region in enumerate(self.vor.regions):
            if -1 not in region and len(region) > 0:
                # Find point that generated this region
                point_idx = np.where((self.vor.point_region == i))[0]
                if len(point_idx) > 0:
                    self.region_labels[i] = self.y_train[point_idx[0]]
                    
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        check_is_fitted(self, ['X_train', 'y_train', 'vor', 'region_labels'])
        X = check_array(X)
        
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            # Find nearest training point
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Get region of nearest point
            region_idx = self.vor.point_region[nearest_idx]
            predictions[i] = self.region_labels.get(region_idx, 0)
            
        return predictions

class PrototypeSelector(Estimator):
    """Intelligent prototype selection for KNN classification."""
    
    def __init__(self, selection_method: Literal['condensed', 'edited'] = 'condensed'):
        self.selection_method = selection_method
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PrototypeSelector':
        """Select prototypes from training data."""
        X, y = check_X_y(X, y)
        
        if self.selection_method == 'condensed':
            self.X_prototypes, self.y_prototypes = self._condensed_selection(X, y)
        elif self.selection_method == 'edited':
            self.X_prototypes, self.y_prototypes = self._edited_selection(X, y)
        else:
            raise ValidationError(f"Unknown selection method: {self.selection_method}")
            
        return self
        
    def _condensed_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Condensed Nearest Neighbor (CNN) selection."""
        prototypes = [X[0]]
        proto_labels = [y[0]]
        
        added = True
        while added:
            added = False
            for x, label in zip(X, y):
                if x in prototypes:
                    continue
                    
                # Find nearest prototype
                distances = np.linalg.norm(prototypes - x, axis=1)
                nearest_label = proto_labels[np.argmin(distances)]
                
                if nearest_label != label:
                    prototypes.append(x)
                    proto_labels.append(label)
                    added = True
                    
        return np.array(prototypes), np.array(proto_labels)
        
    def _edited_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Edited Nearest Neighbor (ENN) selection."""
        mask = np.ones(len(X), dtype=bool)
        k = 3  # Number of neighbors to consider
        
        for i, (x, label) in enumerate(zip(X, y)):
            # Find k nearest neighbors excluding self
            distances = np.linalg.norm(X - x, axis=1)
            distances[i] = float('inf')
            neighbors = np.argsort(distances)[:k]
            neighbor_labels = y[neighbors]
            
            # Remove point if majority of neighbors disagree
            if np.sum(neighbor_labels == label) <= k/2:
                mask[i] = False
                
        return X[mask], y[mask]