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

from core import (
    Layer as BaseLayer, Loss as BaseLoss, Estimator,
    check_array, check_X_y, check_is_fitted,
    EPSILON, DEFAULT_RANDOM_STATE, SUPPORTED_ACTIVATIONS,
    Number, Array, Features, Target
)
from core.logging import get_logger

logger = get_logger(__name__)

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
            scale = np.sqrt(1.0 / input_dim) 
        else:
            scale = 0.01
            
        self.w = scale * np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)
        
        # Initialize gradients
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        self.x = x if training else None
        self.q = self.x @ self.w + self.b
        return self.q
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        batch_size = self.x.shape[0]
        self.dw = self.x.T @ upstream_grad / batch_size
        self.db = np.mean(upstream_grad, axis=0)
        
        # Add regularization gradients
        if 'l2' in self.regularization:
            self.dw += self.regularization['l2'] * self.w
        elif 'l1' in self.regularization:
            self.dw += self.regularization['l1'] * np.sign(self.w)
            
        return upstream_grad @ self.w.T
        
    def get_params(self) -> Dict[str, np.ndarray]:
        return {'w': self.w, 'b': self.b}
        
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self.w = params['w']
        self.b = params['b']

class ReLULayer(Layer):
    """Rectified Linear Unit activation function."""
    
    def __init__(self, alpha: float = 0.0):
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
        return np.where(x > 0, x, self.alpha * x)
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        return upstream_grad * np.where(self.x > 0, 1, self.alpha)

class DropoutLayer(Layer):
    """Dropout regularization layer."""
    
    def __init__(self, drop_rate: float = 0.5):
        super().__init__()
        self.trainable = False
        self.drop_rate = drop_rate
        self.mask = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training
        if training:
            self.mask = np.random.binomial(1, 1-self.drop_rate, x.shape) / (1-self.drop_rate)
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
    """Cross entropy loss with softmax."""
    
    @classmethod
    def forward(cls, logits: np.ndarray, y_true: np.ndarray) -> float:
        cls.y_true = y_true
        # Compute softmax with numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        cls.softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Compute cross entropy
        log_likelihood = -np.log(cls.softmax[range(len(y_true)), y_true])
        cls.loss = np.mean(log_likelihood)
        return cls.loss

    @classmethod
    def backward(cls) -> np.ndarray:
        grad = cls.softmax.copy()
        grad[range(len(cls.y_true)), cls.y_true] -= 1
        return grad / len(cls.y_true)

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
            self.history[metric.__name__] = []
            self.history[f'val_{metric.__name__}'] = []
            
        # Setup optimizer
        self.optimizer = optimizer or {'type': 'sgd', 'learning_rate': 0.01}
        if self.optimizer['type'] == 'adam':
            for layer in self.layers:
                if layer.trainable:
                    layer.m = {k: np.zeros_like(v) for k,v in layer.get_params().items()}
                    layer.v = {k: np.zeros_like(v) for k,v in layer.get_params().items()}

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
        else:  # SGD
            for layer in self.layers:
                if layer.trainable:
                    params = layer.get_params()
                    for k, v in params.items():
                        params[k] -= lr * getattr(layer, f'd{k}')
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