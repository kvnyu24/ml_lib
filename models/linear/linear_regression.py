"""
Linear Regression and Optimization Library
========================================

A comprehensive library for linear regression and optimization:

- Multiple regression algorithms (OLS, Ridge, Lasso, Elastic Net)
- Regularization techniques
- Feature engineering and selection
- Cross-validation and model selection
- Model evaluation and metrics
- Visualization utilities
- Extensible interfaces for custom models

The implementation follows clean design principles with modular components.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core import (
    Estimator, Loss,
    check_array, check_X_y, check_is_fitted,
    get_logger, TrainingLogger,
    Number, Array, Features, Target,
    EPSILON, DEFAULT_RANDOM_STATE
)

from optimization.optimizers import Adam, RMSprop
from ..evaluation import ModelEvaluator

# Configure logging
logger = get_logger(__name__)

class ElasticNetLoss(Loss):
    """Elastic Net loss function with L1 and L2 regularization."""
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Compute loss value.
    
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        y_true = y_true.reshape(-1, 1) if len(y_true.shape) == 1 else y_true
        y_pred = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
        
        # Scale MSE by 0.5 to make gradient simpler
        mse = 0.5 * np.mean((y_true - y_pred) ** 2)
        
        # Add regularization terms if weights are provided
        if weights is not None:
            # Scale down regularization terms
            l1_penalty = 0.01 * self.alpha * self.l1_ratio * np.sum(np.abs(weights))
            l2_penalty = 0.005 * self.alpha * (1 - self.l1_ratio) * np.sum(weights**2)
            return mse + l1_penalty + l2_penalty
        return mse
        
    def gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute gradient of loss with respect to weights.
    
        Args:
            X: Input features
            y_true: Ground truth values
            y_pred: Predicted values
            weights: Model weights
        
        Returns:
            Gradient of loss with respect to predictions
        
        Raises:
            ValueError: If input arrays are empty or have incompatible shapes
        """
        # Input validation
        if X.shape[0] == 0:
            raise ValueError("Empty input array: X must contain at least one sample")
        
        if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
            raise ValueError("Empty target arrays: y_true and y_pred must contain at least one sample")

        # Ensure proper broadcasting by reshaping arrays
        y_true = y_true.reshape(-1, 1) if len(y_true.shape) == 1 else y_true
        y_pred = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred

        n_samples = X.shape[0]
        
        # Gradient of MSE term (simplified due to 0.5 scaling factor)
        grad_mse = -1/n_samples * X.T @ (y_true - y_pred)
        # Scale down regularization gradients
        grad_l1 = 0.01 * self.alpha * self.l1_ratio * np.sign(weights)
        grad_l2 = 0.01 * self.alpha * (1 - self.l1_ratio) * weights

        return grad_mse + grad_l1 + grad_l2

class ElasticNetRegression(Estimator):
    """Elastic Net regression with L1 and L2 regularization."""
    
    def __init__(self,
                 alpha: float = 0.1, # Reduced default regularization strength
                 l1_ratio: float = 0.2, # Reduced L1 ratio to favor L2
                 optimizer: Optional[Adam] = None,
                 fit_intercept: bool = True,
                 max_iter: int = 2000, # Increased iterations
                 tol: float = 1e-6): # Tighter convergence
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        # Use Adam with tuned parameters
        self.optimizer = optimizer or Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.loss = ElasticNetLoss(alpha=alpha, l1_ratio=l1_ratio)
        
    def fit(self, X: Features, y: Target, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        # Validate input data
        X, y = check_X_y(X, y)
        
        if X.shape[0] == 0:
            raise ValueError("Empty input array: X must contain at least one sample")
            
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        self._input_dim = X.shape[1]
        # Initialize weights with small random values
        self.weights = np.random.randn(self._input_dim) * 0.01
        
        # Initialize optimizer with parameter dimensions
        self.optimizer.initialize(self.weights.shape)
        
        prev_loss = float('inf')
        history = {'loss': [], 'val_loss': []}
        
        # Early stopping counter
        patience = 5
        min_loss = float('inf')
        patience_counter = 0
        
        for iter in range(self.max_iter):
            # Forward pass
            y_pred = X @ self.weights
            current_loss = self.loss(y, y_pred, self.weights)
            history['loss'].append(current_loss)
            
            # Validation step
            if validation_data is not None:
                X_val, y_val = validation_data
                if self.fit_intercept:
                    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
                val_pred = X_val @ self.weights
                val_loss = self.loss(y_val, val_pred, self.weights)
                history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < min_loss:
                    min_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at iteration {iter}")
                        break
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tol:
                logger.info(f"Converged at iteration {iter}")
                break
                
            # Compute gradients with correct parameters
            gradients = self.loss.gradient(X, y, y_pred, self.weights)
            self.weights = self.optimizer.compute_update(self.weights, gradients)
            prev_loss = current_loss
            
        self._is_fitted = True
        return history
        
    def predict(self, X: Features) -> Target:
        """Make predictions on new data."""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        return X @ self.weights

class Visualizer:
    """Visualization utilities."""
    
    @staticmethod
    def plot_regression(X: Features,
                       y: Target,
                       y_pred: np.ndarray,
                       title: str = 'Regression Plot') -> None:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Prediction')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def plot_learning_curve(history: Dict[str, List],
                          title: str = 'Learning Curve') -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        plt.show()