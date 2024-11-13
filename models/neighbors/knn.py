"""K-Nearest Neighbors implementations."""

from typing import Literal, Optional, Union, List, Tuple
import numpy as np
from scipy.stats import mode

from core import (
    Estimator,
    check_array,
    check_X_y,
    check_is_fitted,
    ValidationError,
    euclidean_distance
)

class KNeighborsBase(Estimator):
    """Base class for K-Nearest Neighbors algorithms."""
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: str = 'euclidean',
        leaf_size: int = 30
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.leaf_size = leaf_size
        
        # Dictionary of supported distance metrics
        self._metrics = {
            'euclidean': euclidean_distance,
            # Add more metrics as needed
        }
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNeighborsBase':
        """Store the training data."""
        # Validate inputs
        X, y = check_X_y(X, y)
        
        if self.n_neighbors > len(X):
            raise ValidationError(
                f"n_neighbors ({self.n_neighbors}) cannot be larger "
                f"than number of samples ({len(X)})"
            )
            
        self.X_train = X
        self.y_train = y
        return self
        
    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        """Calculate weights based on distances."""
        if self.weights == 'uniform':
            return np.ones(distances.shape)
        elif self.weights == 'distance':
            # Avoid division by zero
            return 1 / (distances + np.finfo(float).eps)
        else:
            raise ValueError(f"Unsupported weight type: {self.weights}")
            
    def _find_neighbors(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k-nearest neighbors for each sample in X."""
        check_is_fitted(self, ['X_train', 'y_train'])
        X = check_array(X)
        
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, len(self.X_train)))
        
        # Calculate distances between X and all training samples
        for i in range(n_samples):
            distances[i] = self._metrics[self.metric](X[i], self.X_train)
            
        # Get indices of k nearest neighbors
        neigh_ind = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        # Get corresponding distances
        neigh_dist = np.take_along_axis(
            distances, neigh_ind, axis=1
        )
        
        return neigh_dist, neigh_ind

class KNeighborsClassifier(KNeighborsBase):
    """K-Nearest Neighbors Classifier."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        neigh_dist, neigh_ind = self._find_neighbors(X)
        
        # Get neighbor labels
        neigh_labels = self.y_train[neigh_ind]
        
        # Calculate weights
        weights = self._get_weights(neigh_dist)
        
        # Weighted voting
        if self.weights == 'uniform':
            # Simple majority voting
            predictions = mode(neigh_labels, axis=1)[0].ravel()
        else:
            # Weighted voting
            predictions = np.zeros(len(X))
            for i in range(len(X)):
                unique_labels = np.unique(neigh_labels[i])
                weighted_votes = np.zeros(len(unique_labels))
                for j, label in enumerate(unique_labels):
                    mask = neigh_labels[i] == label
                    weighted_votes[j] = np.sum(weights[i][mask])
                predictions[i] = unique_labels[np.argmax(weighted_votes)]
                
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        neigh_dist, neigh_ind = self._find_neighbors(X)
        
        # Get neighbor labels
        neigh_labels = self.y_train[neigh_ind]
        
        # Calculate weights
        weights = self._get_weights(neigh_dist)
        
        # Get unique classes
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        
        # Calculate weighted probabilities
        probabilities = np.zeros((len(X), n_classes))
        for i in range(len(X)):
            for j, class_label in enumerate(classes):
                mask = neigh_labels[i] == class_label
                probabilities[i, j] = np.sum(weights[i][mask])
            # Normalize probabilities
            probabilities[i] /= np.sum(probabilities[i])
            
        return probabilities

class KNeighborsRegressor(KNeighborsBase):
    """K-Nearest Neighbors Regressor."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples in X."""
        neigh_dist, neigh_ind = self._find_neighbors(X)
        
        # Get neighbor values
        neigh_values = self.y_train[neigh_ind]
        
        # Calculate weights
        weights = self._get_weights(neigh_dist)
        
        # Weighted average
        weighted_sum = np.sum(weights * neigh_values, axis=1)
        weight_sum = np.sum(weights, axis=1)
        
        return weighted_sum / weight_sum 