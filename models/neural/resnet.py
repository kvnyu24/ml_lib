"""ResNet implementation."""

import numpy as np
from typing import List, Optional, Union, Tuple
from core import Layer, check_array, check_is_fitted, EPSILON
from .layers import BatchNormalization, GlobalAveragePooling2D, Dense
from .cnn import Conv2D

class ResidualBlock(Layer):
    """Basic residual block with two 3x3 convolutions."""
    
    def __init__(self,
                 filters: int,
                 stride: int = 1,
                 activation: str = 'relu',
                 use_projection: bool = False):
        super().__init__()
        self.filters = filters
        self.stride = stride
        self.activation = activation
        self.use_projection = use_projection
        
        # Main path layers
        self.conv1 = Conv2D(filters=filters, 
                           kernel_size=3,
                           stride=stride,
                           padding='same')
        self.bn1 = BatchNormalization()
        
        self.conv2 = Conv2D(filters=filters,
                           kernel_size=3,
                           stride=1,
                           padding='same')
        self.bn2 = BatchNormalization()
        
        # Shortcut path (if needed)
        if use_projection:
            self.proj_conv = Conv2D(filters=filters,
                                  kernel_size=1,
                                  stride=stride)
            self.proj_bn = BatchNormalization()
            
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass computation."""
        # Store input for residual
        identity = x
        
        # Main path
        out = self.conv1(x, training=training)
        out = self.bn1(out, training=training)
        out = self._activate(out)
        
        out = self.conv2(out, training=training)
        out = self.bn2(out, training=training)
        
        # Shortcut path
        if self.use_projection:
            identity = self.proj_conv(x, training=training)
            identity = self.proj_bn(identity, training=training)
            
        # Add residual
        out += identity
        return self._activate(out)
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation."""
        # Store incoming gradient for residual
        d_residual = gradient
        
        # Main path backward
        d_main = self._activate_gradient(gradient)
        d_main = self.bn2.backward(d_main)
        d_main = self.conv2.backward(d_main)
        
        d_main = self._activate_gradient(d_main)
        d_main = self.bn1.backward(d_main)
        d_main = self.conv1.backward(d_main)
        
        # Shortcut path backward
        if self.use_projection:
            d_residual = self.proj_bn.backward(d_residual)
            d_residual = self.proj_conv.backward(d_residual)
            
        return d_main + d_residual
        
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        raise ValueError(f"Unsupported activation: {self.activation}")
        
    def _activate_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute activation gradient."""
        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        raise ValueError(f"Unsupported activation: {self.activation}")

class ResNet(Layer):
    """ResNet architecture."""
    
    def __init__(self,
                 block_counts: List[int],
                 num_classes: int,
                 filters: List[int] = [64, 128, 256, 512],
                 activation: str = 'relu'):
        super().__init__()
        self.block_counts = block_counts
        self.num_classes = num_classes
        self.filters = filters
        self.activation = activation
        
        # Initial convolution
        self.conv1 = Conv2D(filters=64,
                           kernel_size=7,
                           stride=2,
                           padding='same')
        self.bn1 = BatchNormalization()
        
        # Residual blocks
        self.layers = []
        for i, num_blocks in enumerate(block_counts):
            blocks = []
            for j in range(num_blocks):
                use_projection = (j == 0 and i > 0)
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(ResidualBlock(filters=filters[i],
                                         stride=stride,
                                         activation=activation,
                                         use_projection=use_projection))
            self.layers.append(blocks)
            
        # Output layers
        self.global_pool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes)
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass computation."""
        # Initial convolution
        x = self.conv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self._activate(x)
        
        # Residual blocks
        for blocks in self.layers:
            for block in blocks:
                x = block(x, training=training)
                
        # Output
        x = self.global_pool(x)
        return self.fc(x)
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation."""
        dx = self.fc.backward(gradient)
        dx = self.global_pool.backward(dx)
        
        # Residual blocks backward
        for blocks in reversed(self.layers):
            for block in reversed(blocks):
                dx = block.backward(dx)
                
        # Initial layers backward
        dx = self._activate_gradient(dx)
        dx = self.bn1.backward(dx)
        dx = self.conv1.backward(dx)
        
        return dx 