"""Linear Discriminant Analysis implementation."""

import numpy as np
from typing import Optional, Tuple
from core import (
    Estimator,
    check_X_y,
    check_array,
    check_is_fitted,
    EPSILON,
    DEFAULT_RANDOM_STATE
)
from scipy.linalg import eigh

class LDA(Estimator):
    """Linear Discriminant Analysis classifier.
    
    Implements LDA using eigenvalue decomposition for dimensionality 
    reduction and classification.
    """
    
    def __init__(self, n_components: Optional[int] = None, 
                 solver: str = 'eigen',
                 shrinkage: Optional[float] = None):
        """Initialize LDA classifier.
        
        Args:
            n_components: Number of components for dimensionality reduction
            solver: Solver to use ('eigen' or 'svd')
            shrinkage: Shrinkage parameter (None or float between 0 and 1)
        """
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage
        
        # Attributes set during fit
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.scalings_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDA':
        """Fit LDA model.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            self: Fitted estimator
        """
        X, y = check_X_y(X, y)
        
        # Get unique classes and priors
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        if self.n_components is None:
            self.n_components = min(n_features, n_classes - 1)
            
        # Compute class priors
        self.priors_ = np.bincount(y) / n_samples
        
        # Compute means
        self.means_ = np.zeros((n_classes, n_features))
        for i, cls in enumerate(self.classes_):
            self.means_[i] = X[y == cls].mean(axis=0)
            
        # Compute scatter matrices
        Sw = np.zeros((n_features, n_features))  # Within-class scatter
        Sb = np.zeros((n_features, n_features))  # Between-class scatter
        
        # Within-class scatter
        for i, cls in enumerate(self.classes_):
            class_mask = (y == cls)
            n_samples_class = np.sum(class_mask)
            centered_X = X[class_mask] - self.means_[i]
            Sw += centered_X.T @ centered_X
            
        if self.shrinkage is not None:
            # Apply shrinkage to within-class scatter
            shrinkage_target = np.eye(n_features)
            Sw = (1 - self.shrinkage) * Sw + self.shrinkage * shrinkage_target
            
        # Between-class scatter
        overall_mean = np.mean(X, axis=0)
        for i in range(n_classes):
            n_samples_class = np.sum(y == self.classes_[i])
            mean_diff = self.means_[i] - overall_mean
            Sb += n_samples_class * np.outer(mean_diff, mean_diff)
            
        # Solve eigenvalue problem
        if self.solver == 'eigen':
            evals, evecs = eigh(Sb, Sw)
            # Sort eigenvectors by eigenvalues in descending order
            idx = np.argsort(evals)[::-1]
            evecs = evecs[:, idx]
            # Select top n_components eigenvectors
            self.scalings_ = evecs[:, :self.n_components]
            
        # Compute coefficients and intercept for prediction
        self.coef_ = self.scalings_.T @ Sw
        self.intercept_ = -0.5 * np.sum(self.coef_ * self.means_, axis=1)
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X.
        
        Args:
            X: Input data
            
        Returns:
            X_new: Transformed data
        """
        X = check_array(X)
        return X @ self.scalings_
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Class probabilities
        """
        scores = self.decision_function(X)
        # Apply softmax to get probabilities
        exp_scores = np.exp(scores - scores.max(axis=1)[:, np.newaxis])
        probs = exp_scores / exp_scores.sum(axis=1)[:, np.newaxis]
        return probs
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision scores.
        
        Args:
            X: Input data
            
        Returns:
            Decision scores
        """
        X = check_array(X)
        scores = X @ self.coef_.T + self.intercept_
        return scores 