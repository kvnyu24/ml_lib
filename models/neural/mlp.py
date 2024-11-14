"""
Neural Network Library
=====================

This library implements a modular neural network framework with the following components:

- Layer classes for building network architectures
- Loss functions for training
- MLP class for combining layers into a network
- Utility functions for array operations and visualization
- Optimizers for gradient descent variants
- Regularization options
- Model saving/loading
- Training callbacks

The implementation follows modern deep learning practices with clean interfaces
for building and training neural networks.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Tuple, Callable
import pickle
import json
from pathlib import Path
import logging


from core import (
    Layer as BaseLayer, Loss as BaseLoss, Estimator,
    check_array, check_X_y, check_is_fitted,
    EPSILON, DEFAULT_RANDOM_STATE, SUPPORTED_ACTIVATIONS,
    Number, Array, Features, Target
)
from core.logging import get_logger

# Configure root logger to show DEBUG messages


logger = get_logger(__name__)
# Configure the logger directly
logger.setLevel(logging.DEBUG)
# Add a console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class Layer(BaseLayer):
    """Base class for neural network layers."""
    
    def __init__(self):
        super().__init__()
        self.trainable = True
        self.training = True
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass computation.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        raise NotImplementedError
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass computation.
        
        Args:
            upstream_grad: Gradient from the next layer
            
        Returns:
            Gradient with respect to layer input
        """
        raise NotImplementedError
        
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get layer parameters."""
        return {}
        
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set layer parameters."""
        pass

class AffineLayer(Layer):
    """Fully connected layer implementing y = xW + b transformation."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 weight_init: str = 'he', 
                 regularization: Optional[Dict] = None):
        """Initialize layer parameters.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            weight_init: Initialization method ('he', 'xavier', 'normal')
            regularization: Dict with type ('l1', 'l2') and lambda value
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.regularization = regularization or {}
        
        # Initialize weights based on method
        if weight_init == 'he':
            scale = np.sqrt(2.0 / input_dim)
        elif weight_init == 'xavier':
            scale = np.sqrt(2.0 / (input_dim + output_dim))  # Changed to improved Xavier
        else:
            scale = 0.01
            
        # Initialize with better scaling and add bias initialization
        self.w = scale * np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim) + 0.01  # Small positive bias
        
        # Initialize gradients
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        self.x = x if training else None
        # Use faster matrix multiplication
        self.q = np.dot(self.x, self.w) + self.b
        return self.q
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        batch_size = self.x.shape[0]
        # Vectorized operations
        self.dw = np.dot(self.x.T, upstream_grad) / batch_size
        self.db = np.mean(upstream_grad, axis=0)
        
        # Efficient regularization
        if 'l2' in self.regularization:
            lambda_l2 = self.regularization['l2']
            if lambda_l2 > 0:
                self.dw += lambda_l2 * self.w
        elif 'l1' in self.regularization:
            lambda_l1 = self.regularization['l1']
            if lambda_l1 > 0:
                self.dw += lambda_l1 * np.sign(self.w)
            
        return np.dot(upstream_grad, self.w.T)

class ReLULayer(Layer):
    """Rectified Linear Unit activation function."""
    
    def __init__(self, alpha: float = 0.01):  # Changed default to LeakyReLU
        """Initialize ReLU.
        
        Args:
            alpha: Slope for negative values (LeakyReLU)
        """
        super().__init__()
        self.trainable = False
        self.alpha = alpha
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        self.x = x if training else None
        return np.maximum(x, self.alpha * x)  # Faster implementation
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        dx = np.ones_like(self.x)
        dx[self.x < 0] = self.alpha
        return upstream_grad * dx

class DropoutLayer(Layer):
    """Dropout regularization layer."""
    
    def __init__(self, drop_rate: float = 0.2):  # Reduced dropout rate
        super().__init__()
        self.trainable = False
        self.drop_rate = drop_rate
        self.mask = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        if training:
            self.mask = (np.random.rand(*x.shape) > self.drop_rate) / (1-self.drop_rate)
            return x * self.mask
        return x
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        return upstream_grad * self.mask

class Loss(BaseLoss):
    """Base class for loss functions."""
    
    @classmethod
    def forward(cls, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError
        
    @classmethod
    def backward(cls) -> np.ndarray:
        raise NotImplementedError

class QuadraticLoss(Loss):
    """Mean squared error loss."""
    
    @classmethod
    def forward(cls, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        cls.y_pred = y_pred
        cls.y_true = y_true
        cls.loss = 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
        return cls.loss

    @classmethod
    def backward(cls) -> np.ndarray:
        return (cls.y_pred - cls.y_true) / cls.y_pred.shape[0]

class CrossEntropyLoss(Loss):
    @classmethod
    def forward(cls, logits: np.ndarray, y_true: np.ndarray) -> float:
        cls.y_true = y_true
        # Convert one-hot encoded labels to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
            
        # Compute softmax with numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        cls.softmax = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + EPSILON)
        # Compute cross entropy with numerical stability
        cls.loss = np.mean(-np.log(np.maximum(cls.softmax[np.arange(len(y_true)), y_true], EPSILON)))
        return cls.loss

    @classmethod
    def backward(cls) -> np.ndarray:
        grad = cls.softmax.copy()
        # Convert one-hot encoded labels to class indices if needed
        y_true = cls.y_true
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        grad[range(len(y_true)), y_true] -= 1
        return grad / len(y_true)

class MLP(Estimator):
    """Multi-layer perceptron implementation."""
    
    def __init__(self, layers: List[Layer], loss_function: Loss,
                 optimizer: Optional[Dict] = None,
                 metrics: Optional[List[Callable]] = None):
        """Initialize the network.
        
        Args:
            layers: List of Layer objects defining the network
            loss_function: Loss class to use for training
            optimizer: Dict with optimizer config
            metrics: List of metric functions to track
        """
        super().__init__()
        self.layers = layers
        self.loss_function = loss_function
        self.metrics = metrics or []
        self.history = {'loss': [], 'val_loss': []}
        for metric in self.metrics:
            self.history[metric.name] = []
            self.history[f'val_{metric.name}'] = []
            
        # Setup optimizer with better defaults
        self.optimizer = optimizer or {
            'type': 'adam',
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        }
        
        if self.optimizer['type'] == 'adam':
            for layer in self.layers:
                if layer.trainable:
                    layer.m = {k: np.zeros_like(v) for k,v in layer.get_params().items()}
                    layer.v = {k: np.zeros_like(v) for k,v in layer.get_params().items()}

        self._is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            batch_size: int = 128, epochs: int = 10,  # Increased batch size
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'MLP':
        """Fit the neural network to training data.
        
        Args:
            X: Training features
            y: Target values
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train
            validation_data: Tuple of (X_val, y_val) for validation
            
        Returns:
            self: The fitted model
        """
        logger.debug("Starting fit method")
        logger.info("Starting model training")
        logger.warning("This is a test warning")
        logger.debug(f"Input shapes - X: {X.shape}, y: {y.shape}")
        
        X, y = check_X_y(X, y)
        self._input_dim = X.shape[1]
        self._output_dim = y.shape[1] if len(y.shape) > 1 else 1
        logger.debug(f"Set input_dim={self._input_dim}, output_dim={self._output_dim}")
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Training
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            
            for batch in range(n_batches):
                batch_idx = indices[batch*batch_size:(batch+1)*batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                loss = self.forward(X_batch, y_batch)
                epoch_loss += loss
                
                # Backward pass
                self.backward()
                
                # Update parameters
                self.update_params(epoch * n_batches + batch)
                
            epoch_loss /= n_batches
            self.history['loss'].append(epoch_loss)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.forward(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                
                # Compute metrics
                y_pred = self.predict(X_val)
                for metric in self.metrics:
                    score = metric(y_val, y_pred)
                    self.history[f'val_{metric.name}'].append(score)
                    
            logger.info(f'Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}' + 
                       (f' - val_loss: {val_loss:.4f}' if validation_data else ''))
        
        self._is_fitted = True
        logger.debug(f"Set is_fitted={self._is_fitted}")
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Model predictions
        """
        logger.debug("Starting predict method")
        logger.debug(f"Model fitted status: {getattr(self, '_is_fitted', False)}")
        # check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self._input_dim:
            raise ValueError(f"Expected {self._input_dim} features, got {X.shape[1]}")
            
        return self.forward(X)

    def forward(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
        """Forward pass through the network."""
        self.activations = []
        a = x
        for layer in self.layers:
            a = layer.forward(a, training=(y is not None))
            self.activations.append(a)
        self.y_pred = a
        
        if y is not None:
            self.loss = self.loss_function.forward(a, y)
            return self.loss
        return self.y_pred

    def backward(self) -> None:
        """Backward pass to compute gradients."""
        grad = self.loss_function.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
    def update_params(self, iteration: int) -> None:
        """Update parameters using configured optimizer."""
        lr = self.optimizer['learning_rate']
        if self.optimizer['type'] == 'adam':
            beta1 = self.optimizer.get('beta1', 0.9)
            beta2 = self.optimizer.get('beta2', 0.999)
            eps = self.optimizer.get('epsilon', EPSILON)
            
            for layer in self.layers:
                if layer.trainable:
                    params = layer.get_params()
                    for k, v in params.items():
                        grad = getattr(layer, f'd{k}')
                        layer.m[k] = beta1 * layer.m[k] + (1-beta1) * grad
                        layer.v[k] = beta2 * layer.v[k] + (1-beta2) * grad**2
                        m_hat = layer.m[k] / (1 - beta1**(iteration+1))
                        v_hat = layer.v[k] / (1 - beta2**(iteration+1))
                        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                    layer.set_params(params)
        else:  # SGD with momentum
            momentum = self.optimizer.get('momentum', 0.9)
            for layer in self.layers:
                if layer.trainable:
                    if not hasattr(layer, 'velocity'):
                        layer.velocity = {k: np.zeros_like(v) for k,v in layer.get_params().items()}
                    params = layer.get_params()
                    for k, v in params.items():
                        grad = getattr(layer, f'd{k}')
                        layer.velocity[k] = momentum * layer.velocity[k] - lr * grad
                        params[k] += layer.velocity[k]
                    layer.set_params(params)
                    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model parameters and architecture."""
        model_data = {
            'architecture': [layer.__class__.__name__ for layer in self.layers],
            'parameters': [layer.get_params() for layer in self.layers],
            'optimizer': self.optimizer,
            'history': self.history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MLP':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        # Reconstruct model...
        return cls

def array_string(a: np.ndarray) -> str:
    """Convert array to formatted string representation."""
    if np.isscalar(a) or a.ndim == 0:
        return '{:.4g}'.format(a)
    return '(' + ', '.join([array_string(x) for x in a]) + ')'

def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute dot product handling scalar and tensor inputs."""
    return a * b if np.isscalar(a) or np.isscalar(b) else np.einsum('...i,i...', a, b)