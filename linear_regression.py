"""
Linear Regression and Optimization Library
========================================

A comprehensive library for linear regression and optimization:

- Multiple regression algorithms (OLS, Ridge, Lasso, Elastic Net)
- Advanced optimization methods (SGD, Adam, RMSprop, Momentum)
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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    @abstractmethod
    def minimize(self,
                objective: Callable,
                initial_params: np.ndarray,
                **kwargs) -> Tuple[np.ndarray, float, Dict]:
        """Minimize objective function."""
        pass

class AdamOptimizer(BaseOptimizer):
    """Adam optimizer with momentum and adaptive learning rates."""
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 max_iter: int = 1000):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter
        
    def minimize(self,
                objective: Callable,
                initial_params: np.ndarray,
                **kwargs) -> Tuple[np.ndarray, float, Dict]:
        params = initial_params.copy()
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment
        history = {'loss': [], 'params': []}
        
        for t in range(1, self.max_iter + 1):
            grad = self._compute_gradient(objective, params)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2
            
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            
            params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            loss = objective(params)
            history['loss'].append(loss)
            history['params'].append(params.copy())
            
        return params, objective(params), history

class RMSpropOptimizer(BaseOptimizer):
    """RMSprop optimizer with adaptive learning rates."""
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 decay_rate: float = 0.9,
                 epsilon: float = 1e-8,
                 max_iter: int = 1000):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        
    def minimize(self,
                objective: Callable,
                initial_params: np.ndarray,
                **kwargs) -> Tuple[np.ndarray, float, Dict]:
        params = initial_params.copy()
        cache = np.zeros_like(params)
        history = {'loss': [], 'params': []}
        
        for _ in range(self.max_iter):
            grad = self._compute_gradient(objective, params)
            cache = self.decay_rate * cache + (1 - self.decay_rate) * grad**2
            params -= self.learning_rate * grad / (np.sqrt(cache) + self.epsilon)
            
            loss = objective(params)
            history['loss'].append(loss)
            history['params'].append(params.copy())
            
        return params, objective(params), history

class ElasticNetRegression(BaseEstimator, RegressorMixin):
    """Elastic Net regression with L1 and L2 regularization."""
    
    def __init__(self,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 optimizer: Optional[BaseOptimizer] = None,
                 fit_intercept: bool = True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.optimizer = optimizer or AdamOptimizer()
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetRegression':
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        def loss(params):
            mse = np.mean((X @ params - y) ** 2)
            l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(params))
            l2_penalty = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(params**2)
            return mse + l1_penalty + l2_penalty
            
        initial_params = np.zeros(X.shape[1])
        params, _, _ = self.optimizer.minimize(loss, initial_params)
        
        if self.fit_intercept:
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            self.coef_ = params
            
        return self

class ModelSelector:
    """Model selection and hyperparameter tuning."""
    
    @staticmethod
    def cross_validate(model: BaseEstimator,
                      X: np.ndarray,
                      y: np.ndarray,
                      k_folds: int = 5) -> Dict[str, float]:
        kf = KFold(n_splits=k_folds, shuffle=True)
        metrics = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            metrics['mse'].append(mean_squared_error(y_val, y_pred))
            metrics['mae'].append(mean_absolute_error(y_val, y_pred))
            metrics['r2'].append(r2_score(y_val, y_pred))
            
        return {k: np.mean(v) for k, v in metrics.items()}

class ModelEvaluator:
    """Model evaluation and metrics."""
    
    @staticmethod
    def evaluate_regression(model: BaseEstimator,
                          X: np.ndarray,
                          y: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model."""
        y_pred = model.predict(X)
        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

class Visualizer:
    """Visualization utilities."""
    
    @staticmethod
    def plot_regression(X: np.ndarray,
                       y: np.ndarray,
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