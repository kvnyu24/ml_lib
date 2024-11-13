"""Experimental graph neural network layers."""

import numpy as np
from typing import Optional, Dict, Union, Callable
from core import (
    Layer,
    EPSILON,
    DEFAULT_RANDOM_STATE
)
from scipy import sparse
import warnings

class GraphConvolution(Layer):
    """Graph convolutional layer with support for different normalization schemes and activations.
    
    Implements the graph convolution operation from Kipf & Welling (2017):
    https://arxiv.org/abs/1609.02907
    """
    
    SUPPORTED_ACTIVATIONS = {
        'relu': lambda x: np.maximum(0, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': np.tanh,
        'leaky_relu': lambda x, alpha=0.01: np.where(x > 0, x, alpha * x)
    }
    
    def __init__(self,
                 in_features: int,
                 out_features: int, 
                 activation: Optional[Union[str, Callable]] = 'relu',
                 dropout: float = 0.0,
                 use_bias: bool = True,
                 weight_init: str = 'glorot_uniform',
                 norm_type: str = 'symmetric'):
        """Initialize the layer.
        
        Args:
            in_features: Number of input features per node
            out_features: Number of output features per node
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'leaky_relu' or callable)
            dropout: Dropout rate between 0 and 1
            use_bias: Whether to use bias term
            weight_init: Weight initialization scheme ('glorot_uniform', 'glorot_normal', 'he')
            norm_type: Graph normalization type ('symmetric', 'left', 'right')
        """
        super().__init__()
        
        if not 0 <= dropout < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
            
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.use_bias = use_bias
        self.norm_type = norm_type
        
        # Set activation function
        if isinstance(activation, str):
            self.activation_fn = self.SUPPORTED_ACTIVATIONS.get(activation.lower())
            if self.activation_fn is None:
                raise ValueError(f"Unsupported activation: {activation}")
        elif callable(activation):
            self.activation_fn = activation
        else:
            self.activation_fn = None
            
        # Initialize weights
        self.W = self._initialize_weights(weight_init)
        self.b = np.zeros(out_features) if use_bias else None
        
        # For gradient updates
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if use_bias else None
        
        # Cache for backward pass
        self.cache = {}
        
    def _initialize_weights(self, init_type: str) -> np.ndarray:
        """Initialize weights using different schemes."""
        if init_type == 'glorot_uniform':
            limit = np.sqrt(6 / (self.in_features + self.out_features))
            return np.random.uniform(-limit, limit, (self.in_features, self.out_features))
        elif init_type == 'glorot_normal':
            std = np.sqrt(2 / (self.in_features + self.out_features))
            return np.random.normal(0, std, (self.in_features, self.out_features))
        elif init_type == 'he':
            std = np.sqrt(2 / self.in_features)
            return np.random.normal(0, std, (self.in_features, self.out_features))
        else:
            raise ValueError(f"Unsupported weight initialization: {init_type}")
    
    def _normalize_adj(self, A: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix based on normalization type."""
        # Add self-loops
        A_hat = A + np.eye(A.shape[0])
        
        # Calculate degree matrix
        D = np.sum(A_hat, axis=1)
        D_inv = 1 / (D + EPSILON)
        D_inv_sqrt = np.sqrt(D_inv)
        
        if self.norm_type == 'symmetric':
            D_inv_sqrt = np.diag(D_inv_sqrt)
            return D_inv_sqrt @ A_hat @ D_inv_sqrt
        elif self.norm_type == 'left':
            D_inv = np.diag(D_inv)
            return D_inv @ A_hat
        elif self.norm_type == 'right':
            D_inv = np.diag(D_inv)
            return A_hat @ D_inv
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")
            
    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Forward pass with adjacency matrix A.
        
        Args:
            X: Node features matrix of shape (num_nodes, in_features)
            A: Adjacency matrix of shape (num_nodes, num_nodes)
            
        Returns:
            Output features of shape (num_nodes, out_features)
        """
        if X.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {X.shape[1]}")
            
        # Apply dropout to input
        if self.dropout > 0 and self.training:
            mask = np.random.binomial(1, 1-self.dropout, X.shape) / (1-self.dropout)
            X = X * mask
            
        # Normalize adjacency matrix
        A_norm = self._normalize_adj(A)
        
        # Cache for backward pass
        self.cache['X'] = X
        self.cache['A_norm'] = A_norm
        
        # Graph convolution operation
        Z = A_norm @ X @ self.W
        if self.use_bias:
            Z = Z + self.b
            
        # Apply activation
        if self.activation_fn is not None:
            return self.activation_fn(Z)
        return Z