"""Activation function implementations."""

import numpy as np
from core import Layer, EPSILON
from typing import Dict

class ReLU(Layer):
    """ReLU activation layer."""
    
    def __init__(self, alpha: float = 0.0):
        """Initialize ReLU layer.
        
        Args:
            alpha: Slope for negative values (LeakyReLU if > 0)
        """
        super().__init__()
        self.alpha = alpha
        self.x = None
        self.trainable = False
        
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {'alpha': np.array([self.alpha])}
        
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters."""
        if 'alpha' in params:
            self.alpha = float(params['alpha'][0])
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass applying ReLU activation."""
        self.x = x if training else None
        return np.where(x > 0, x, self.alpha * x)
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients."""
        return upstream_grad * np.where(self.x > 0, 1, self.alpha)

class Sigmoid(Layer):
    """Sigmoid activation layer."""
    
    def __init__(self):
        super().__init__()
        self.output = None
        self.trainable = False
        
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {'output': self.output} if self.output is not None else {}
        
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters."""
        if 'output' in params:
            self.output = params['output']
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass applying sigmoid activation."""
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients."""
        return upstream_grad * self.output * (1 - self.output)

class Tanh(Layer):
    """Hyperbolic tangent activation layer."""
    
    def __init__(self):
        super().__init__()
        self.output = None
        self.trainable = False
        
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {'output': self.output} if self.output is not None else {}
        
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters."""
        if 'output' in params:
            self.output = params['output']
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass applying tanh activation."""
        self.output = np.tanh(x)
        return self.output
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients."""
        return upstream_grad * (1 - self.output ** 2)

class Softmax(Layer):
    """Softmax activation layer."""
    
    def __init__(self):
        super().__init__()
        self.output = None
        self.trainable = False
        
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {'output': self.output} if self.output is not None else {}
        
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set trainable parameters."""
        if 'output' in params:
            self.output = params['output']
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass applying softmax activation."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients."""
        # Jacobian matrix of softmax times upstream gradient
        return self.output * (upstream_grad - np.sum(upstream_grad * self.output, axis=-1, keepdims=True))

def get_activation(name: str) -> Layer:
    """Get activation layer by name."""
    if name == 'relu':
        return ReLU()
    elif name == 'sigmoid':
        return Sigmoid() 
    elif name == 'tanh':
        return Tanh()
    elif name == 'softmax':
        return Softmax()
    else:
        raise ValueError(f"Unknown activation function: {name}")