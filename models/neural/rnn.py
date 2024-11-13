"""Recurrent Neural Network implementations."""

import numpy as np
from typing import Optional, Union, Tuple, Dict
from core import (
    Layer,
    check_array,
    check_is_fitted,
    EPSILON,
    DEFAULT_RANDOM_STATE
)

class RNNCell(Layer):
    """Basic RNN cell implementation."""
    
    def __init__(self,
                 units: int,
                 activation: str = 'tanh',
                 use_bias: bool = True,
                 return_sequences: bool = False,
                 kernel_initializer: str = 'glorot_uniform',
                 recurrent_initializer: str = 'orthogonal'):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.return_sequences = return_sequences
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        
        # Will be initialized during build
        self.kernel = None  # Input weights
        self.recurrent_kernel = None  # Hidden state weights
        self.bias = None
        self.states = None
        
    def build(self, input_shape: tuple) -> None:
        """Initialize layer parameters."""
        input_dim = input_shape[-1]
        
        # Initialize input weights
        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6 / (input_dim + self.units))
            self.kernel = np.random.uniform(-limit, limit, (input_dim, self.units))
        else:
            self.kernel = np.random.randn(input_dim, self.units) * 0.01
            
        # Initialize recurrent weights
        if self.recurrent_initializer == 'orthogonal':
            self.recurrent_kernel = self._generate_orthogonal_matrix(self.units)
        else:
            limit = np.sqrt(6 / (self.units * 2))
            self.recurrent_kernel = np.random.uniform(-limit, limit, (self.units, self.units))
            
        if self.use_bias:
            self.bias = np.zeros(self.units)
            
    def _generate_orthogonal_matrix(self, n: int) -> np.ndarray:
        """Generate random orthogonal matrix."""
        random_matrix = np.random.randn(n, n)
        q, r = np.linalg.qr(random_matrix)
        return q * np.sign(np.diag(r))
        
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass computation."""
        batch_size, time_steps, input_dim = inputs.shape
        self.inputs = inputs
        
        # Initialize hidden states
        h = np.zeros((batch_size, self.units))
        hidden_states = np.zeros((batch_size, time_steps, self.units))
        
        # Process sequence
        for t in range(time_steps):
            # Current input
            x_t = inputs[:, t, :]
            
            # Compute hidden state
            h = self._activation(
                np.dot(x_t, self.kernel) +
                np.dot(h, self.recurrent_kernel) +
                (self.bias if self.use_bias else 0)
            )
            hidden_states[:, t, :] = h
            
        self.states = hidden_states
        return hidden_states if self.return_sequences else hidden_states[:, -1, :]
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation."""
        batch_size, time_steps, _ = self.inputs.shape
        
        if not self.return_sequences:
            # Expand gradient for all timesteps
            full_gradient = np.zeros((batch_size, time_steps, self.units))
            full_gradient[:, -1, :] = gradient
            gradient = full_gradient
            
        # Initialize gradients
        dx = np.zeros_like(self.inputs)
        dkernel = np.zeros_like(self.kernel)
        drecurrent = np.zeros_like(self.recurrent_kernel)
        if self.use_bias:
            dbias = np.zeros(self.units)
            
        # Initialize hidden state gradient
        dh_next = np.zeros((batch_size, self.units))
        
        # Backpropagate through time
        for t in reversed(range(time_steps)):
            # Current gradients
            dh = gradient[:, t, :] + dh_next
            
            # Input gradients
            dx[:, t, :] = np.dot(dh, self.kernel.T)
            
            # Weight gradients
            dkernel += np.dot(self.inputs[:, t, :].T, dh)
            drecurrent += np.dot(
                self.states[:, t-1, :].T if t > 0 
                else np.zeros((batch_size, self.units)).T, 
                dh
            )
            
            if self.use_bias:
                dbias += np.sum(dh, axis=0)
                
            # Gradient for next timestep
            dh_next = np.dot(dh, self.recurrent_kernel.T)
            
        # Store gradients
        self.kernel_gradient = dkernel
        self.recurrent_gradient = drecurrent
        if self.use_bias:
            self.bias_gradient = dbias
            
        return dx
        
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}") 