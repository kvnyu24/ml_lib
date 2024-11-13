"""Core interfaces and base classes for ML library."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import logging
from .validation import check_array, check_X_y, check_is_fitted

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Estimator(ABC):
    """Base class for all estimators."""
    
    def __init__(self):
        self._is_fitted = False
        self._input_dim = None
        self._output_dim = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'Estimator':
        """Fit the estimator to training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted estimator
            
        Raises:
            ValueError: If input dimensions are invalid
            TypeError: If input types are invalid
        """
        X, y = check_X_y(X, y)
        self._input_dim = X.shape[1] if len(X.shape) > 1 else 1
        if y is not None:
            self._output_dim = y.shape[1] if len(y.shape) > 1 else 1
        self._is_fitted = True
        return self
    
    @abstractmethod 
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on, shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
            
        Raises:
            RuntimeError: If estimator is not fitted
            ValueError: If input dimensions don't match training data
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self._input_dim:
            raise ValueError(f"Expected {self._input_dim} features, got {X.shape[1]}")
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters.
        
        Returns:
            Dict of parameter names and values
        """
        params = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                params[k] = v
        return params
    
    def set_params(self, **params) -> 'Estimator':
        """Set estimator parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self
            
        Raises:
            ValueError: If invalid parameter name provided
        """
        valid_params = self.get_params()
        for k, v in params.items():
            if k not in valid_params:
                raise ValueError(f"Invalid parameter {k}")
            setattr(self, k, v)
        return self
        
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate input arrays.
        
        Args:
            X: Input features
            y: Optional target values
            
        Raises:
            TypeError: If inputs are not numpy arrays
            ValueError: If input shapes are incompatible
        """
        if y is not None:
            X, y = check_X_y(X, y)
        else:
            X = check_array(X)

class Optimizer(ABC):
    """Base class for optimization algorithms."""
    
    def __init__(self, learning_rate: float = 0.01):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        self.learning_rate = learning_rate
        self._iteration = 0
        self._state = {}
    
    @abstractmethod
    def compute_update(self, params: np.ndarray, 
                      gradients: np.ndarray) -> np.ndarray:
        """Compute parameter updates.
        
        Args:
            params: Current parameter values
            gradients: Parameter gradients
            
        Returns:
            Updated parameter values
            
        Raises:
            ValueError: If params and gradients shapes don't match
        """
        self._iteration += 1
        self._validate_inputs(params, gradients)
        return params
    
    def reset_state(self) -> None:
        """Reset optimizer state."""
        self._iteration = 0
        self._state = {}
        
    def _validate_inputs(self, params: np.ndarray, gradients: np.ndarray) -> None:
        """Validate optimizer inputs.
        
        Args:
            params: Parameter array
            gradients: Gradient array
            
        Raises:
            ValueError: If arrays have different shapes
            TypeError: If inputs are not numpy arrays
        """
        params = check_array(params)
        gradients = check_array(gradients)
        if params.shape != gradients.shape:
            raise ValueError("params and gradients must have same shape")

class Loss(ABC):
    """Base class for loss functions."""
    
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss value.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Loss value
            
        Raises:
            ValueError: If input shapes don't match
        """
        self._validate_inputs(y_true, y_pred)
        return 0.0
    
    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute loss gradient.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Loss gradient
            
        Raises:
            ValueError: If input shapes don't match
        """
        self._validate_inputs(y_true, y_pred)
        return np.zeros_like(y_true)
        
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Validate loss function inputs.
        
        Args:
            y_true: Ground truth array
            y_pred: Predictions array
            
        Raises:
            ValueError: If arrays have different shapes
            TypeError: If inputs are not numpy arrays
        """
        y_true = check_array(y_true)
        y_pred = check_array(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have same shape")

class Layer(ABC):
    """Base class for neural network layers."""
    
    def __init__(self):
        self.trainable = True
        self.weights = None
        self.gradients = None
        self._input_shape = None
        self._output_shape = None
        self._initialized = False
    
    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass computation.
        
        Args:
            inputs: Layer inputs
            training: Whether in training mode
            
        Returns:
            Layer outputs
            
        Raises:
            ValueError: If input shape is invalid
        """
        if self._input_shape is None:
            self._input_shape = inputs.shape[1:]
            self._initialized = True
        elif inputs.shape[1:] != self._input_shape:
            raise ValueError(f"Expected input shape {self._input_shape}, got {inputs.shape[1:]}")
        return inputs
    
    @abstractmethod
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Backward pass computation.
        
        Args:
            gradients: Upstream gradients
            
        Returns:
            Downstream gradients
            
        Raises:
            RuntimeError: If called before forward pass
        """
        if not self._initialized:
            raise RuntimeError("Layer must perform forward pass before backward pass")
        if not self.trainable:
            return gradients
        return gradients
        
    def get_weights(self) -> List[np.ndarray]:
        """Get layer weights.
        
        Returns:
            List of weight arrays
        """
        return [self.weights] if self.weights is not None else []
        
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set layer weights.
        
        Args:
            weights: List of weight arrays
            
        Raises:
            ValueError: If weights list is empty when layer has weights
        """
        if self.weights is not None and len(weights) == 0:
            raise ValueError("Weights list cannot be empty for layer with weights")
        if len(weights) > 0:
            self.weights = weights[0]

class BaseTransformer(ABC):
    """Base class for all transformers.
    
    All transformers should inherit from this class and implement:
        - fit
        - transform
    
    The fit method is used to learn parameters from training data.
    The transform method applies the transformation using learned parameters.
    """
    
    def __init__(self):
        self._is_fitted = False
        self._feature_names_in = None
        self._n_features_in = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseTransformer':
        """Learn transformation parameters from training data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,). Optional for unsupervised transformers.
            
        Returns:
            self: The fitted transformer
            
        Raises:
            ValueError: If input validation fails
        """
        X = check_array(X)
        self._is_fitted = True
        return self
        
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply transformation using learned parameters.
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            X_transformed: Transformed data
            
        Raises:
            NotFittedError: If transformer is not fitted
            ValueError: If input validation fails
        """
        check_is_fitted(self)
        X = check_array(X)
        return X
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit transformer and apply transformation.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,). Optional for unsupervised transformers.
            
        Returns:
            X_transformed: Transformed data
        """
        return self.fit(X, y).transform(X)
        
    def _validate_input(self, X: np.ndarray) -> None:
        """Validate input data.
        
        Args:
            X: Input data to validate
            
        Raises:
            ValueError: If validation fails
        """
        X = check_array(X)
            
        if not self._is_fitted:
            self._n_features_in = X.shape[1] if X.ndim > 1 else 1
            
        elif X.ndim > 1 and X.shape[1] != self._n_features_in:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"was fitted with {self._n_features_in} features"
            )
            
    def _check_is_fitted(self) -> None:
        """Check if transformer is fitted.
        
        Raises:
            NotFittedError: If transformer is not fitted
        """
        check_is_fitted(self)
            
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
        
    def set_params(self, **params) -> 'BaseTransformer':
        """Set transformer parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
    def __repr__(self) -> str:
        """String representation of transformer."""
        params = [f"{k}={v}" for k, v in self.get_params().items()]
        return f"{self.__class__.__name__}({', '.join(params)})"