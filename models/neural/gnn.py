"""Graph neural network implementations."""

import numpy as np
from typing import Optional, Union, Callable
from core import (
    Layer,
    EPSILON,
    DEFAULT_RANDOM_STATE
)

class GraphConvolution(Layer):
    """Graph convolutional layer.
    
    Implements the graph convolution operation from Kipf & Welling (2017):
    https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int, 
                 activation: Optional[str] = 'relu',
                 dropout: float = 0.1,
                 use_bias: bool = True,
                 norm_type: str = 'symmetric'):
        """Initialize the layer.
        
        Args:
            in_features: Number of input features per node
            out_features: Number of output features per node
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            dropout: Dropout rate between 0 and 1
            use_bias: Whether to use bias term
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
        self.activation = activation
        
        # Will be initialized during build
        self.W = None
        self.b = None
        
    def build(self, input_shape: tuple) -> None:
        """Initialize layer parameters."""
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (self.in_features + self.out_features))
        self.W = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
        
        if self.use_bias:
            self.b = np.zeros(self.out_features)
    
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
        if not hasattr(self, 'W'):
            self.build(X.shape)
            
        if X.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {X.shape[1]}")
            
        # Apply dropout to input
        if self.dropout > 0 and self.training:
            mask = np.random.binomial(1, 1-self.dropout, X.shape) / (1-self.dropout)
            X = X * mask
            
        # Normalize adjacency matrix
        A_norm = self._normalize_adj(A)
        
        # Graph convolution operation
        Z = A_norm @ X @ self.W
        if self.use_bias:
            Z = Z + self.b
            
        # Apply activation
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation == 'tanh':
            return np.tanh(Z)
        return Z