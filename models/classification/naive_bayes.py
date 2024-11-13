"""Implementation of Naive Bayes classifier."""

import numpy as np
from typing import Dict, Optional
from .base import BaseClassifier
from core import (
    get_logger,
    check_array,
    check_X_y,
    check_is_fitted,
    ValidationError
)

logger = get_logger(__name__)

class NaiveBayesClassifier(BaseClassifier):
    """Gaussian Naive Bayes classifier implementation."""
    
    def __init__(self):
        self.class_priors = None
        self.means = None 
        self.variances = None
        self.classes = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Gaussian Naive Bayes classifier.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        X, y = check_X_y(X, y)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)
        
        # Calculate class priors and feature statistics
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i, :] = X_c.mean(axis=0)
            self.variances[i, :] = X_c.var(axis=0)
            self.class_priors[i] = X_c.shape[0] / X.shape[0]
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ['means', 'variances', 'class_priors'])
        X = check_array(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))
        
        # Calculate log probabilities for each class
        for i in range(n_classes):
            # Gaussian likelihood
            exponent = -0.5 * np.sum(
                (X - self.means[i, :]) ** 2 / self.variances[i, :], axis=1
            )
            normalizer = -0.5 * np.sum(np.log(2 * np.pi * self.variances[i, :]))
            log_probs = exponent + normalizer + np.log(self.class_priors[i])
            probs[:, i] = log_probs
            
        # Normalize probabilities
        probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
