"""Convolutional Neural Network implementation."""

import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from core import (
    Layer,
    check_array,
    check_is_fitted,
    EPSILON,
    DEFAULT_RANDOM_STATE
)

class Conv2D(Layer):
    """2D convolutional layer."""
    
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides: Union[int, Tuple[int, int]] = (1, 1),
                 padding: str = 'valid',
                 activation: Optional[str] = None,
                 use_bias: bool = True):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        
        # Will be initialized during build
        self.kernels = None
        self.bias = None
        
    def build(self, input_shape: tuple) -> None:
        """Initialize layer parameters."""
        channels = input_shape[-1]
        kernel_shape = (*self.kernel_size, channels, self.filters)
        
        # Xavier initialization for kernels
        limit = np.sqrt(6 / (np.prod(self.kernel_size) * channels + self.filters))
        self.kernels = np.random.uniform(-limit, limit, kernel_shape)
        
        if self.use_bias:
            self.bias = np.zeros(self.filters)
            
    def _pad_input(self, inputs: np.ndarray) -> np.ndarray:
        """Apply padding to input."""
        if self.padding == 'valid':
            return inputs
            
        if self.padding == 'same':
            h_pad = (self.kernel_size[0] - 1) // 2
            w_pad = (self.kernel_size[1] - 1) // 2
            return np.pad(inputs, 
                         ((0, 0), (h_pad, h_pad), (w_pad, w_pad), (0, 0)),
                         mode='constant')
        
        raise ValueError(f"Unsupported padding type: {self.padding}")
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass computation."""
        self.inputs = inputs
        batch_size, in_h, in_w, _ = inputs.shape
        
        # Apply padding
        padded = self._pad_input(inputs)
        
        # Calculate output dimensions
        out_h = (padded.shape[1] - self.kernel_size[0]) // self.strides[0] + 1
        out_w = (padded.shape[2] - self.kernel_size[1]) // self.strides[1] + 1
        
        # Initialize output
        outputs = np.zeros((batch_size, out_h, out_w, self.filters))
        
        # Perform convolution
        for i in range(0, out_h):
            for j in range(0, out_w):
                h_start = i * self.strides[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.strides[1]
                w_end = w_start + self.kernel_size[1]
                
                receptive_field = padded[:, h_start:h_end, w_start:w_end, :]
                outputs[:, i, j, :] = np.tensordot(receptive_field, 
                                                 self.kernels,
                                                 axes=[(1, 2, 3), (0, 1, 2)])
        
        if self.use_bias:
            outputs += self.bias
            
        if self.activation:
            outputs = self._apply_activation(outputs)
            
        return outputs
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation."""
        if self.activation:
            gradient = self._apply_activation_gradient(gradient)
            
        batch_size = gradient.shape[0]
        padded = self._pad_input(self.inputs)
        
        # Initialize gradients
        kernel_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(padded)
        
        # Calculate gradients
        for i in range(gradient.shape[1]):
            for j in range(gradient.shape[2]):
                h_start = i * self.strides[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.strides[1]
                w_end = w_start + self.kernel_size[1]
                
                receptive_field = padded[:, h_start:h_end, w_start:w_end, :]
                
                # Kernel gradients
                for b in range(batch_size):
                    kernel_gradient += np.einsum('ijc,f->ijcf', 
                                               receptive_field[b],
                                               gradient[b, i, j])
                
                # Input gradients
                input_gradient[:, h_start:h_end, w_start:w_end] += \
                    np.einsum('bijk,klm->bijm', 
                             gradient[:, i:i+1, j:j+1, :],
                             np.transpose(self.kernels, (0, 1, 3, 2)))
        
        if self.use_bias:
            self.bias_gradient = np.sum(gradient, axis=(0, 1, 2))
            
        # Remove padding from input gradient if needed
        if self.padding == 'same':
            h_pad = (self.kernel_size[0] - 1) // 2
            w_pad = (self.kernel_size[1] - 1) // 2
            input_gradient = input_gradient[:, h_pad:-h_pad, w_pad:-w_pad, :]
            
        return input_gradient 