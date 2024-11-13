"""Common neural network layers implementation."""

import numpy as np
from typing import Optional, Union, Tuple, Callable
from core import Layer, EPSILON

class Dense(Layer):
    """Fully connected layer."""
    
    def __init__(self,
                 units: int,
                 activation: Optional[str] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform'):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        
        # Will be initialized during build
        self.W = None
        self.b = None
        
    def build(self, input_shape: tuple) -> None:
        """Initialize layer parameters."""
        input_dim = input_shape[-1]
        
        # Xavier/Glorot initialization
        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6 / (input_dim + self.units))
            self.W = np.random.uniform(-limit, limit, (input_dim, self.units))
        elif self.kernel_initializer == 'glorot_normal':
            std = np.sqrt(2 / (input_dim + self.units))
            self.W = np.random.normal(0, std, (input_dim, self.units))
            
        if self.use_bias:
            self.b = np.zeros(self.units)
            
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass computation."""
        if not hasattr(self, 'W'):
            self.build(x.shape)
            
        z = x @ self.W
        if self.use_bias:
            z = z + self.b
            
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        return z

class Dropout(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, rate: float):
        super().__init__()
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        self.rate = rate
        self.mask = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with dropout."""
        if not training or self.rate == 0:
            return x
            
        self.mask = np.random.binomial(1, 1-self.rate, x.shape) / (1-self.rate)
        return x * self.mask

class Flatten(Layer):
    """Flattens input while preserving batch size."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Flatten input tensor."""
        return x.reshape(x.shape[0], -1)

class MaxPool2D(Layer):
    """2D Max Pooling layer."""
    
    def __init__(self, pool_size: Union[int, Tuple[int, int]] = (2, 2),
                 strides: Optional[Union[int, Tuple[int, int]]] = None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with max pooling."""
        batch_size, height, width, channels = x.shape
        
        # Calculate output dimensions
        out_height = 1 + (height - self.pool_size[0]) // self.strides[0]
        out_width = 1 + (width - self.pool_size[1]) // self.strides[1]
        
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        for b in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.strides[0]
                    h_end = h_start + self.pool_size[0]
                    w_start = j * self.strides[1]
                    w_end = w_start + self.pool_size[1]
                    window = x[b, h_start:h_end, w_start:w_end, :]
                    output[b, i, j, :] = np.max(window, axis=(0, 1))
                    
        return output

class GlobalAveragePooling2D(Layer):
    """Global Average Pooling 2D layer."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with global average pooling."""
        return np.mean(x, axis=(1, 2))


class BatchNormalization(Layer):
    """Batch normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = np.mean(x, axis=(0, 1, 2))
            var = np.var(x, axis=(0, 1, 2))
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta