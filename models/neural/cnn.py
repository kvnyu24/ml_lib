"""Convolutional Neural Network implementation with ResNet capabilities."""

import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Callable
from core import (
    Layer,
    check_array,
    check_is_fitted,
    EPSILON,
    DEFAULT_RANDOM_STATE
)
from .layers import BatchNormalization

class Conv2D(Layer):
    """2D convolutional layer with advanced features."""
    
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 strides: Union[int, Tuple[int, int]] = (1, 1),
                 padding: str = 'valid',
                 activation: Optional[str] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 kernel_regularizer: Optional[Callable] = None,
                 groups: int = 1):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.lower()
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.groups = groups
        
        if self.groups <= 0:
            raise ValueError("groups must be a positive integer")
        
        # Will be initialized during build
        self.kernels = None
        self.bias = None
        
    def build(self, input_shape: tuple) -> None:
        """Initialize layer parameters."""
        channels = input_shape[-1]
        
        if channels % self.groups != 0:
            raise ValueError("input channels must be divisible by groups")
        if self.filters % self.groups != 0:
            raise ValueError("filters must be divisible by groups")
            
        kernel_shape = (*self.kernel_size, channels // self.groups, self.filters)
        
        # Initialize kernels based on specified initializer
        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6 / (np.prod(self.kernel_size) * channels + self.filters))
            self.kernels = np.random.uniform(-limit, limit, kernel_shape)
        elif self.kernel_initializer == 'he_normal':
            std = np.sqrt(2 / (np.prod(self.kernel_size) * channels))
            self.kernels = np.random.normal(0, std, kernel_shape)
        else:
            raise ValueError(f"Unsupported initializer: {self.kernel_initializer}")
        
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
        """Forward pass computation with grouped convolutions."""
        if not hasattr(self, 'kernels'):
            self.build(inputs.shape)
            
        self.inputs = inputs
        batch_size, in_h, in_w, in_c = inputs.shape
        
        if in_c != self.kernels.shape[2] * self.groups:
            raise ValueError(f"Input channels ({in_c}) doesn't match kernel channels ({self.kernels.shape[2] * self.groups})")
        
        # Apply padding
        padded = self._pad_input(inputs)
        
        # Calculate output dimensions
        out_h = (padded.shape[1] - self.kernel_size[0]) // self.strides[0] + 1
        out_w = (padded.shape[2] - self.kernel_size[1]) // self.strides[1] + 1
        
        # Initialize output
        outputs = np.zeros((batch_size, out_h, out_w, self.filters))
        
        # Split input and kernels into groups
        grouped_inputs = np.split(padded, self.groups, axis=-1)
        grouped_kernels = np.split(self.kernels, self.groups, axis=-1)
        
        # Perform grouped convolutions
        for group in range(self.groups):
            for i in range(0, out_h):
                for j in range(0, out_w):
                    h_start = i * self.strides[0]
                    h_end = h_start + self.kernel_size[0]
                    w_start = j * self.strides[1]
                    w_end = w_start + self.kernel_size[1]
                    
                    receptive_field = grouped_inputs[group][:, h_start:h_end, w_start:w_end, :]
                    outputs[:, i, j, group::self.groups] = np.tensordot(
                        receptive_field,
                        grouped_kernels[group],
                        axes=[(1, 2, 3), (0, 1, 2)]
                    )
        
        if self.use_bias:
            outputs += self.bias
            
        if self.activation:
            outputs = self._apply_activation(outputs)
            
        # Apply kernel regularization if specified
        if self.kernel_regularizer is not None:
            self.add_loss(self.kernel_regularizer(self.kernels))
            
        return outputs
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation with grouped convolutions."""
        if self.activation:
            gradient = self._apply_activation_gradient(gradient)
            
        batch_size = gradient.shape[0]
        padded = self._pad_input(self.inputs)
        
        # Initialize gradients
        kernel_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(padded)
        
        # Split gradients and inputs into groups
        grouped_gradients = np.split(gradient, self.groups, axis=-1)
        grouped_inputs = np.split(padded, self.groups, axis=-1)
        grouped_kernels = np.split(self.kernels, self.groups, axis=-1)
        
        # Calculate gradients for each group
        for group in range(self.groups):
            for i in range(gradient.shape[1]):
                for j in range(gradient.shape[2]):
                    h_start = i * self.strides[0]
                    h_end = h_start + self.kernel_size[0]
                    w_start = j * self.strides[1]
                    w_end = w_start + self.kernel_size[1]
                    
                    receptive_field = grouped_inputs[group][:, h_start:h_end, w_start:w_end, :]
                    
                    # Kernel gradients
                    for b in range(batch_size):
                        kernel_gradient[:, :, :, group::self.groups] += np.einsum(
                            'ijc,f->ijcf',
                            receptive_field[b],
                            grouped_gradients[group][b, i, j]
                        )
                    
                    # Input gradients
                    input_gradient[:, h_start:h_end, w_start:w_end, group::self.groups] += \
                        np.einsum('bijk,klm->bijm',
                                 grouped_gradients[group][:, i:i+1, j:j+1, :],
                                 np.transpose(grouped_kernels[group], (0, 1, 3, 2)))
        
        if self.use_bias:
            self.bias_gradient = np.sum(gradient, axis=(0, 1, 2))
            
        # Remove padding from input gradient if needed
        if self.padding == 'same':
            h_pad = (self.kernel_size[0] - 1) // 2
            w_pad = (self.kernel_size[1] - 1) // 2
            input_gradient = input_gradient[:, h_pad:-h_pad, w_pad:-w_pad, :]
            
        return input_gradient