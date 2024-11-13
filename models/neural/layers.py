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
        self.x = None
        self.z = None
        
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
            
        self.x = x
        self.z = x @ self.W
        if self.use_bias:
            self.z = self.z + self.b
            
        if self.activation == 'relu':
            return np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-self.z))
        elif self.activation == 'tanh':
            return np.tanh(self.z)
        return self.z

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass computation."""
        # Apply activation gradient if needed
        if self.activation == 'relu':
            upstream_grad = upstream_grad * (self.z > 0)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-self.z))
            upstream_grad = upstream_grad * s * (1 - s)
        elif self.activation == 'tanh':
            upstream_grad = upstream_grad * (1 - np.tanh(self.z)**2)

        # Compute gradients
        self.dW = self.x.T @ upstream_grad / self.x.shape[0]
        if self.use_bias:
            self.db = np.mean(upstream_grad, axis=0)
            
        # Return gradient with respect to input
        return upstream_grad @ self.W.T

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
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass for dropout."""
        if self.mask is None:
            return upstream_grad
        return upstream_grad * self.mask

class Flatten(Layer):
    """Flattens input while preserving batch size."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Flatten input tensor."""
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Reshape gradient back to input shape."""
        return upstream_grad.reshape(self.input_shape)

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
        self.x = x
        batch_size, height, width, channels = x.shape
        
        # Calculate output dimensions
        out_height = 1 + (height - self.pool_size[0]) // self.strides[0]
        out_width = 1 + (width - self.pool_size[1]) // self.strides[1]
        
        output = np.zeros((batch_size, out_height, out_width, channels))
        self.max_indices = np.zeros_like(x)
        
        for b in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.strides[0]
                    h_end = h_start + self.pool_size[0]
                    w_start = j * self.strides[1]
                    w_end = w_start + self.pool_size[1]
                    window = x[b, h_start:h_end, w_start:w_end, :]
                    
                    # Store indices of max values for backprop
                    window_flat = window.reshape(-1, channels)
                    max_idx = np.argmax(window_flat, axis=0)
                    h_idx = h_start + max_idx // window.shape[1]
                    w_idx = w_start + max_idx % window.shape[1]
                    self.max_indices[b, h_idx, w_idx, range(channels)] = 1
                    
                    output[b, i, j, :] = np.max(window, axis=(0, 1))
                    
        return output
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass for max pooling."""
        grad = np.zeros_like(self.x)
        batch_size, out_height, out_width, channels = upstream_grad.shape
        
        for b in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.strides[0]
                    h_end = h_start + self.pool_size[0]
                    w_start = j * self.strides[1]
                    w_end = w_start + self.pool_size[1]
                    
                    # Route gradient to max locations
                    mask = self.max_indices[b, h_start:h_end, w_start:w_end, :]
                    grad[b, h_start:h_end, w_start:w_end, :] += mask * upstream_grad[b, i, j, :]
                    
        return grad

class GlobalAveragePooling2D(Layer):
    """Global Average Pooling 2D layer."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with global average pooling."""
        self.input_shape = x.shape
        return np.mean(x, axis=(1, 2))
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass for global average pooling."""
        # Distribute gradient evenly across spatial dimensions
        grad = np.expand_dims(np.expand_dims(upstream_grad, 1), 1)
        return np.ones(self.input_shape) * grad / (self.input_shape[1] * self.input_shape[2])

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
        self.x = x
        if training:
            self.mean = np.mean(x, axis=(0, 1, 2))
            self.var = np.var(x, axis=(0, 1, 2))
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var
            
        # Normalize
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta
        
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backward pass for batch normalization."""
        N = upstream_grad.shape[0] * upstream_grad.shape[1] * upstream_grad.shape[2]
        
        # Gradients with respect to gamma and beta
        self.dgamma = np.sum(upstream_grad * self.x_norm, axis=(0,1,2))
        self.dbeta = np.sum(upstream_grad, axis=(0,1,2))
        
        # Gradient with respect to input
        dx_norm = upstream_grad * self.gamma
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * (self.var + self.eps)**(-1.5), axis=(0,1,2))
        dmean = np.sum(dx_norm * -1/np.sqrt(self.var + self.eps), axis=(0,1,2)) + dvar * np.mean(-2 * (self.x - self.mean), axis=(0,1,2))
        
        dx = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mean) / N + dmean / N
        return dx