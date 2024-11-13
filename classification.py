"""
Classification and Model Selection Library
========================================

A comprehensive library for classification, model selection and evaluation:

- Data generation and preprocessing
- Multiple classification algorithms (Softmax, SVM, Random Forest, Neural Networks)
- Advanced optimization methods (SGD, Adam, RMSprop)
- Model selection and validation
- Ensemble methods and boosting
- Feature selection and engineering
- Bootstrap and cross-validation analysis
- Performance metrics and visualization
- Extensible interfaces for custom models

The implementation follows clean design principles with modular components.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Dataset:
    """Container for dataset with features and labels."""
    X: np.ndarray  # Features
    y: np.ndarray  # Labels
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    
    def split(self, test_size: float = 0.2, val_size: float = 0.2,
             random_state: int = 42) -> Tuple['Dataset', 'Dataset', 'Dataset']:
        """Split into train, validation and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, 
            random_state=random_state, stratify=self.y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size,
            random_state=random_state, stratify=y_train
        )
        
        return (Dataset(X_train, y_train, self.feature_names, self.target_names),
                Dataset(X_val, y_val, self.feature_names, self.target_names),
                Dataset(X_test, y_test, self.feature_names, self.target_names))
                
    def select_features(self, k: int = 10) -> 'Dataset':
        """Select top k features using mutual information."""
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(self.X, self.y)
        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        return Dataset(X_new, self.y, selected_features, self.target_names)

class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""
    
    @abstractmethod
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using gradients."""
        pass

class AdamOptimizer(BaseOptimizer):
    """Adam optimizer with momentum and adaptive learning rates."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class BaseClassifier(ABC):
    """Abstract base class for classifiers."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier."""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate classifier performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X) if hasattr(self, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_proba, multi_class='ovr')
            
        return metrics

class SoftmaxClassifier(BaseClassifier):
    """Softmax classifier with regularization and custom optimizer."""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 optimizer: Optional[BaseOptimizer] = None):
        self.C = C
        self.max_iter = max_iter
        self.optimizer = optimizer or AdamOptimizer()
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, multi_class='multinomial'
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class SVMClassifier(BaseClassifier):
    """SVM classifier with kernel methods."""
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale'):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class EnsembleClassifier(BaseClassifier):
    """Ensemble classifier combining multiple base models."""
    
    def __init__(self, models: List[BaseClassifier], weights: Optional[np.ndarray] = None):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for model in self.models:
            model.fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict_proba(X) for model in self.models])
        weighted_pred = np.sum(predictions * self.weights[:, np.newaxis, np.newaxis], axis=0)
        return np.argmax(weighted_pred, axis=1)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict_proba(X) for model in self.models])
        return np.sum(predictions * self.weights[:, np.newaxis, np.newaxis], axis=0)

class ModelSelector:
    """Model selection using bootstrap, cross-validation and ensemble methods."""
    
    def __init__(self, 
                 model_class: type,
                 param_grid: Dict[str, List[Any]],
                 n_bootstrap: int = 100,
                 cv_folds: int = 5):
        self.model_class = model_class
        self.param_grid = param_grid
        self.n_bootstrap = n_bootstrap
        self.cv_folds = cv_folds
        
    def bootstrap_select(self, dataset: Dataset) -> Tuple[Dict[str, Any], float]:
        """Select best parameters using bootstrap."""
        best_params = None
        best_score = -np.inf
        
        n_samples = len(dataset.y)
        for params in self._param_combinations():
            scores = []
            for _ in range(self.n_bootstrap):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = dataset.X[indices]
                y_boot = dataset.y[indices]
                
                model = self.model_class(**params)
                model.fit(X_boot, y_boot)
                score = model.evaluate(dataset.X, dataset.y)['accuracy']
                scores.append(score)
                
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                
        return best_params, best_score
        
    def cross_validate(self, dataset: Dataset, params: Dict[str, Any]) -> Dict[str, float]:
        """Perform cross-validation for given parameters."""
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True)
        metrics = defaultdict(list)
        
        for train_idx, val_idx in kf.split(dataset.X, dataset.y):
            X_train, X_val = dataset.X[train_idx], dataset.X[val_idx]
            y_train, y_val = dataset.y[train_idx], dataset.y[val_idx]
            
            model = self.model_class(**params)
            model.fit(X_train, y_train)
            scores = model.evaluate(X_val, y_val)
            
            for metric, value in scores.items():
                metrics[metric].append(value)
                
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

class Visualizer:
    """Visualization utilities for classification."""
    
    @staticmethod
    def plot_decision_boundary(model: BaseClassifier,
                             X: np.ndarray,
                             y: np.ndarray,
                             resolution: float = 0.02) -> None:
        """Plot 2D decision boundary."""
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )
        
        Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.show()
        
    @staticmethod
    def plot_learning_curve(model: BaseClassifier,
                          dataset: Dataset,
                          train_sizes: np.ndarray,
                          cv_folds: int = 5) -> None:
        """Plot learning curve with cross-validation."""
        train_scores = []
        val_scores = []
        
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True)
        
        for size in train_sizes:
            train_fold_scores = []
            val_fold_scores = []
            
            for train_idx, val_idx in kf.split(dataset.X, dataset.y):
                X_train, X_val = dataset.X[train_idx], dataset.X[val_idx]
                y_train, y_val = dataset.y[train_idx], dataset.y[val_idx]
                
                # Subsample training data
                indices = np.random.choice(len(y_train), size, replace=False)
                X_subset = X_train[indices]
                y_subset = y_train[indices]
                
                model.fit(X_subset, y_subset)
                train_score = model.evaluate(X_subset, y_subset)['accuracy']
                val_score = model.evaluate(X_val, y_val)['accuracy']
                
                train_fold_scores.append(train_score)
                val_fold_scores.append(val_score)
            
            train_scores.append(np.mean(train_fold_scores))
            val_scores.append(np.mean(val_fold_scores))
            
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', label='Training score')
        plt.plot(train_sizes, val_scores, 'o-', label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()