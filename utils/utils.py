"""
Machine Learning Utilities Library
================================

A comprehensive library implementing advanced machine learning utilities and algorithms:

Core Functionality:
- Data loading, preprocessing and augmentation
- Model evaluation, cross-validation and metrics
- Feature selection and dimensionality reduction
- Hyperparameter optimization
- Model persistence and serialization

Optimization:
- Gradient descent variants (SGD, Adam, RMSprop)
- Line search and trust region methods
- Constrained optimization
- Convex optimization solvers

Advanced Mechanisms:
- Online/incremental learning
- Active learning
- Transfer learning
- Meta-learning
- Ensemble methods
- Automated ML

Visualization and Analysis:
- Learning curves and validation curves
- Feature importance plots
- Model interpretation tools
- Performance metrics visualization
- Decision boundaries

The implementation follows clean design principles with extensible interfaces.
"""

# Core imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import pickle
import logging
from enum import Enum
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Small constant to prevent division by zero

# ============= Advanced Mechanisms =============

class ActiveLearner:
    """Active learning implementation."""
    
    def __init__(self, model: BaseEstimator, query_strategy: str = 'uncertainty'):
        self.model = model
        self.query_strategy = query_strategy
        
    def query(self, X_pool: np.ndarray, n_instances: int = 1) -> np.ndarray:
        """Select most informative instances for labeling."""
        if self.query_strategy == 'uncertainty':
            probas = self.model.predict_proba(X_pool)
            uncertainties = 1 - np.max(probas, axis=1)
            return np.argsort(uncertainties)[-n_instances:]
        return np.random.choice(len(X_pool), n_instances, replace=False)

class TransferLearner:
    """Transfer learning implementation."""
    
    def __init__(self, base_model: BaseEstimator, target_model: BaseEstimator):
        self.base_model = base_model
        self.target_model = target_model
        
    def transfer_knowledge(self, X_source: np.ndarray, y_source: np.ndarray,
                         X_target: np.ndarray, y_target: np.ndarray):
        """Transfer knowledge from source to target domain."""
        self.base_model.fit(X_source, y_source)
        source_features = self.base_model.transform(X_target)
        self.target_model.fit(source_features, y_target)

class MetaLearner:
    """Meta-learning implementation."""
    
    def __init__(self, base_models: List[BaseEstimator]):
        self.base_models = base_models
        self.meta_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train meta-learner on base model predictions."""
        base_predictions = np.column_stack([
            model.fit(X, y).predict_proba(X)[:, 1] 
            for model in self.base_models
        ])
        self.meta_model = LogisticRegression().fit(base_predictions, y)

class EnsembleLearner:
    """Ensemble learning implementation."""
    
    def __init__(self, models: List[BaseEstimator], weights: Optional[np.ndarray] = None):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all models in ensemble."""
        for model in self.models:
            model.fit(X, y)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted prediction from all models."""
        predictions = np.array([model.predict_proba(X)[:, 1] for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)

class AutoML:
    """Automated machine learning implementation."""
    
    def __init__(self, models: List[BaseEstimator], 
                 param_distributions: List[Dict],
                 max_time: int = 3600):
        self.models = models
        self.param_distributions = param_distributions
        self.max_time = max_time
        self.best_model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Automatically select and tune best model."""
        best_score = float('-inf')
        start_time = time.time()
        
        for model, params in zip(self.models, self.param_distributions):
            if time.time() - start_time > self.max_time:
                break
                
            search = RandomizedSearchCV(model, params, n_jobs=-1)
            search.fit(X, y)
            
            if search.best_score_ > best_score:
                best_score = search.best_score_
                self.best_model = search.best_estimator_

# ============= Impurity Measures =============

class SplitCriterion(Enum):
    """Enumeration of splitting criteria."""
    GINI = "gini"
    ENTROPY = "entropy" 
    ERROR = "error"
    VARIANCE = "variance"

class ImpurityMeasure(ABC):
    """Abstract base class for impurity measures used in decision trees."""
    
    @abstractmethod
    def calculate(self, p: np.ndarray) -> np.ndarray:
        """Calculate impurity for given probability distribution."""
        pass
    
    @abstractmethod
    def gradient(self, p: np.ndarray) -> np.ndarray:
        """Calculate gradient of impurity measure."""
        pass

class ErrorRate(ImpurityMeasure):
    """Misclassification error rate impurity measure."""
    
    def calculate(self, p: np.ndarray) -> np.ndarray:
        q = 1 - p
        return 2 * np.minimum(p + EPSILON, q + EPSILON)
    
    def gradient(self, p: np.ndarray) -> np.ndarray:
        return -2 * np.sign(p - 0.5)

class GiniIndex(ImpurityMeasure):
    """Gini impurity index measure."""
    
    def calculate(self, p: np.ndarray) -> np.ndarray:
        q = 1 - p
        return 4 * (p + EPSILON) * (q + EPSILON)
    
    def gradient(self, p: np.ndarray) -> np.ndarray:
        return 4 * (1 - 2*p)

class Entropy(ImpurityMeasure):
    """Information entropy impurity measure."""
    
    def calculate(self, p: np.ndarray) -> np.ndarray:
        q = 1 - p
        p_safe = p + EPSILON
        q_safe = q + EPSILON
        entropy = -p_safe * np.log2(p_safe) - q_safe * np.log2(q_safe)
        return np.where((p <= 0) | (q <= 0), 0, entropy)
    
    def gradient(self, p: np.ndarray) -> np.ndarray:
        q = 1 - p
        return np.log2(q + EPSILON) - np.log2(p + EPSILON)

class Variance(ImpurityMeasure):
    """Variance impurity measure for regression trees."""
    
    def calculate(self, p: np.ndarray) -> np.ndarray:
        q = 1 - p
        return 4 * (p + EPSILON) * (q + EPSILON)
    
    def gradient(self, p: np.ndarray) -> np.ndarray:
        return 4 * (1 - 2*p)

# ============= Optimizers =============

class Optimizer(ABC):
    """Base class for optimization algorithms."""
    
    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform one optimization step."""
        pass

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity - self.learning_rate * grads
        return params + self.velocity

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None
        
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.cache is None:
            self.cache = np.zeros_like(params)
        
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * np.square(grads)
        return params - self.learning_rate * grads / (np.sqrt(self.cache) + self.epsilon)

# ============= Visualization Tools =============

class ImpurityVisualizer:
    """Advanced visualization tools for impurity measures."""
    
    def __init__(self, measures: Dict[str, ImpurityMeasure]):
        self.measures = measures
        self.colors = {
            'error_rate': '#1f77b4',
            'gini': '#2ca02c',
            'entropy': '#d62728',
            'variance': '#9467bd'
        }
        
    def plot_measures(self, p: np.ndarray, show_gradients: bool = False,
                     save_path: Optional[str] = None) -> None:
        """Plot multiple impurity measures for comparison."""
        fig, (ax1, ax2) if show_gradients else (ax1,) = plt.subplots(
            2 if show_gradients else 1, 1, 
            figsize=(10, 10 if show_gradients else 6),
            sharex=True
        )
        
        for name, measure in self.measures.items():
            style = '--' if name == 'variance' else '-'
            color = self.colors.get(name, 'gray')
            
            values = measure.calculate(p)
            ax1.plot(p, values, label=f'Normalized {name.title()}',
                    color=color, linestyle=style, linewidth=2)
            
            if show_gradients:
                grads = measure.gradient(p)
                ax2.plot(p, grads, label=f'{name.title()} Gradient',
                        color=color, linestyle=style, linewidth=2)
        
        ax1.set_title('Normalized Impurity Measures')
        ax1.set_ylabel('Normalized Impurity')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(0, 1.1)
        
        if show_gradients:
            ax2.set_title('Impurity Measure Gradients')
            ax2.set_xlabel('p (Probability of Positive Class)')
            ax2.set_ylabel('Gradient')
            ax2.legend()
            ax2.grid(True)
        else:
            ax1.set_xlabel('p (Probability of Positive Class)')
            
        plt.xlim(0, 1)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

# ============= Data Structures =============

@dataclass 
class BoundingBox:
    """2D bounding box representation with validation."""
    left: float
    right: float
    bottom: float
    top: float
    
    def __post_init__(self):
        if self.left >= self.right:
            raise ValueError("Left coordinate must be less than right")
        if self.bottom >= self.top:
            raise ValueError("Bottom coordinate must be less than top")
    
    def replace(self, **kwargs) -> 'BoundingBox':
        new_attrs = self.__dict__.copy()
        new_attrs.update(kwargs)
        return BoundingBox(**new_attrs)
    
    @property
    def width(self) -> float:
        return self.right - self.left
    
    @property 
    def height(self) -> float:
        return self.top - self.bottom
    
    @property
    def area(self) -> float:
        return self.width * self.height

# ============= Data Loading and Processing =============

class DataLoader:
    """Advanced data loading and preprocessing utilities."""
    
    @staticmethod
    def load_pickle(path: Union[str, Path], validate: bool = True) -> Any:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        logger.info(f"Loading pickle file: {path}")
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            if validate:
                if not isinstance(data, (np.ndarray, dict, list)):
                    raise ValueError("Invalid data format")
                
            return data
            
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            raise
            
    @staticmethod
    def save_pickle(obj: Any, path: Union[str, Path],
                   compress: bool = False) -> None:
        path = Path(path)
        logger.info(f"Saving pickle file: {path}")
        
        try:
            if compress:
                joblib.dump(obj, path, compress=3)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            logger.error(f"Error saving pickle file: {e}")
            raise

# ============= Model Selection and Evaluation =============

class ModelSelector:
    """Model selection and hyperparameter optimization."""
    
    def __init__(self, model: BaseEstimator, param_distributions: Dict):
        self.model = model
        self.param_distributions = param_distributions
        
    def select_best_model(self, X: np.ndarray, y: np.ndarray, 
                         n_iter: int = 10, cv: int = 5) -> BaseEstimator:
        search = RandomizedSearchCV(
            self.model, self.param_distributions,
            n_iter=n_iter, cv=cv, n_jobs=-1
        )
        search.fit(X, y)
        return search.best_estimator_

# ============= Feature Selection =============

class FeatureSelector:
    """Feature selection utilities."""
    
    @staticmethod
    def select_k_best(X: np.ndarray, y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        feature_scores = selector.scores_
        return X_selected, feature_scores

# ============= Online Learning =============

class OnlineLearner:
    """Online/incremental learning implementation."""
    
    def __init__(self, model: BaseEstimator, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        for i in range(0, len(X), self.batch_size):
            batch_X = X[i:i + self.batch_size]
            batch_y = y[i:i + self.batch_size]
            self.model.partial_fit(batch_X, batch_y)