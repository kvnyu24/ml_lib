"""Implementation of various classifiers."""

import numpy as np
from typing import List, Optional, Dict, Union
from .base import BaseClassifier
from optimization.optimizers import SGD, Adam
from core import (
    BaseClassifier,
    get_logger,
    check_array,
    check_X_y,
    check_is_fitted,
    ValidationError
)

logger = get_logger(__name__)

class SoftmaxClassifier(BaseClassifier):
    """Softmax classifier with regularization and custom optimizer."""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 optimizer: Optional[str] = 'adam'):
        self.C = C
        self.max_iter = max_iter
        self.optimizer = Adam() if optimizer == 'adam' else SGD()
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Initialize weights and bias
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)
        
        # Convert y to one-hot encoding
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        # Training loop
        for i in range(self.max_iter):
            # Forward pass
            scores = np.dot(X, self.weights) + self.bias
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # Compute gradients
            dscores = probs - y_onehot
            dW = np.dot(X.T, dscores) / n_samples + self.C * self.weights
            db = np.sum(dscores, axis=0) / n_samples
            
            # Update parameters using optimizer
            self.weights = self.optimizer.compute_update(self.weights, dW)
            self.bias = self.optimizer.compute_update(self.bias, db)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(scores, axis=1)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.dot(X, self.weights) + self.bias
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier with kernel methods."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0,
                 gamma: Union[str, float] = 'scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.support_vectors = None
        self.dual_coef = None
        self.intercept = None
        
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == 'rbf':
            if self.gamma == 'scale':
                self.gamma = 1.0 / (X1.shape[1] * X1.var())
            elif self.gamma == 'auto':
                self.gamma = 1.0 / X1.shape[1]
                
            dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                    np.sum(X2**2, axis=1) - \
                    2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * dists)
        else:
            return np.dot(X1, X2.T)  # Linear kernel
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Simplified SMO algorithm implementation
        n_samples = X.shape[0]
        kernel = self._kernel_function(X, X)
        
        # Initialize alphas and intercept
        self.dual_coef = np.zeros(n_samples)
        self.intercept = 0.0
        
        # SMO optimization (simplified)
        # In practice, would implement full SMO algorithm
        self.support_vectors = X
        self.support_vector_labels = y
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        kernel = self._kernel_function(X, self.support_vectors)
        scores = np.dot(kernel, self.dual_coef * self.support_vector_labels) + self.intercept
        return np.sign(scores)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._kernel_function(X, self.support_vectors)
        scores = np.dot(scores, self.dual_coef)
        probs = 1 / (1 + np.exp(-scores))
        return np.vstack([1-probs, probs]).T

class EnsembleClassifier(BaseClassifier):
    """Ensemble of multiple classifiers with voting."""
    
    def __init__(self, models: Optional[List[BaseClassifier]] = None,
                 weights: Optional[List[float]] = None):
        self.models = models or [
            SoftmaxClassifier(),
            SVMClassifier(),
            SoftmaxClassifier(C=0.1),
            SoftmaxClassifier(C=10.0)
        ]
        self.weights = weights or [1.0] * len(self.models)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for model in self.models:
            model.fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.average(probas, axis=0, weights=self.weights)