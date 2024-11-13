"""Neural network layer implementations."""

import numpy as np
from typing import Optional, Union, Callable
from core import Layer, check_array, EPSILON
from core.dtypes import SUPPORTED_ACTIVATIONS

class Dense(Layer):
    """Fully connected layer implementation."""
    
    def __init__(self, 
                 units: int,
                 activation: Optional[str] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros'):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        # Will be set when layer is built
        self.kernel = None
        self.bias = None
        self.input_shape = None
        
    def build(self, input_shape: tuple) -> None:
        """Initialize layer weights."""
        self.input_shape = input_shape
        input_dim = input_shape[-1]
        
        # Initialize kernel
        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6 / (input_dim + self.units))
            self.kernel = np.random.uniform(-limit, limit, (input_dim, self.units))
        else:
            self.kernel = np.random.randn(input_dim, self.units) * 0.01
            
        # Initialize bias
        if self.use_bias:
            self.bias = np.zeros(self.units) if self.bias_initializer == 'zeros' \
                       else np.random.randn(self.units) * 0.01
                       
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass computation."""
        outputs = np.dot(inputs, self.kernel)
        if self.use_bias:
            outputs = outputs + self.bias
            
        if self.activation:
            outputs = self._apply_activation(outputs)
            
        return outputs
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation."""
        if self.activation:
            gradient = self._apply_activation_gradient(gradient)
            
        # Compute gradients
        self.kernel_gradient = np.dot(self.inputs.T, gradient)
        if self.use_bias:
            self.bias_gradient = np.sum(gradient, axis=0)
            
        # Propagate gradient
        return np.dot(gradient, self.kernel.T) 