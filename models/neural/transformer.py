"""Transformer architecture implementation."""

import numpy as np
from typing import Optional, Union, Tuple, List
from core import (
    Layer,
    check_array,
    check_is_fitted,
    EPSILON,
    DEFAULT_RANDOM_STATE
)

class MultiHeadAttention(Layer):
    """Multi-head self-attention mechanism."""
    
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.causal = causal
        
        # Will be initialized during build
        self.w_q = None  # Query weights
        self.w_k = None  # Key weights
        self.w_v = None  # Value weights
        self.w_o = None  # Output weights
        
    def build(self, input_shape: tuple) -> None:
        """Initialize layer parameters."""
        # Xavier initialization
        limit = np.sqrt(6 / (self.d_model * 2))
        
        # Initialize attention weights
        self.w_q = np.random.uniform(-limit, limit, (self.d_model, self.d_model))
        self.w_k = np.random.uniform(-limit, limit, (self.d_model, self.d_model))
        self.w_v = np.random.uniform(-limit, limit, (self.d_model, self.d_model))
        self.w_o = np.random.uniform(-limit, limit, (self.d_model, self.d_model))
        
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (num_heads, d_k)."""
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine the heads back into original shape."""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass computation."""
        batch_size = q.shape[0]
        
        # Store inputs for backward pass
        self.q, self.k, self.v = q, k, v
        
        # Linear transformations
        q = np.dot(q, self.w_q)  # (batch_size, seq_len, d_model)
        k = np.dot(k, self.w_k)  # (batch_size, seq_len, d_model)
        v = np.dot(v, self.w_v)  # (batch_size, seq_len, d_model)
        
        # Split heads
        q = self._split_heads(q)  # (batch_size, num_heads, seq_len, d_k)
        k = self._split_heads(k)  # (batch_size, num_heads, seq_len, d_k)
        v = self._split_heads(v)  # (batch_size, num_heads, seq_len, d_k)
        
        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Apply causal mask if needed
        if self.causal:
            seq_len = q.shape[2]
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = np.where(causal_mask, -np.inf, scores)
            
        # Apply attention mask if provided
        if mask is not None:
            scores = np.where(mask, -np.inf, scores)
            
        # Attention weights
        self.attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attention_weights /= np.sum(self.attention_weights, axis=-1, keepdims=True) + EPSILON
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            self.dropout_mask = np.random.binomial(1, 1-self.dropout, self.attention_weights.shape)
            self.attention_weights *= self.dropout_mask
            self.attention_weights /= (1 - self.dropout)
            
        # Compute attention output
        attention_output = np.matmul(self.attention_weights, v)
        
        # Combine heads and apply output transformation
        output = self._combine_heads(attention_output)
        output = np.dot(output, self.w_o)
        
        return output
        
    def backward(self, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass computation."""
        batch_size = gradient.shape[0]
        
        # Output weights gradient
        d_output = gradient
        self.w_o_gradient = np.dot(
            self._combine_heads(self.attention_output).transpose(0, 2, 1),
            d_output
        ).mean(axis=0)
        
        # Split gradient for attention computation
        d_output = self._split_heads(np.dot(d_output, self.w_o.T))
        
        # Attention weights gradient
        if self.dropout > 0 and self.training:
            d_output *= self.dropout_mask
            
        d_scores = np.matmul(d_output, self.v.transpose(0, 1, 3, 2))
        d_attention_weights = d_scores * self.attention_weights
        d_attention_weights -= self.attention_weights * np.sum(
            d_attention_weights, axis=-1, keepdims=True
        )
        
        # Value weights gradient
        d_v = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), d_output)
        self.w_v_gradient = np.dot(
            self.v.transpose(0, 2, 1),
            self._combine_heads(d_v)
        ).mean(axis=0)
        
        # Key weights gradient
        d_k = np.matmul(d_attention_weights.transpose(0, 1, 3, 2), 
                       self.q) / np.sqrt(self.d_k)
        self.w_k_gradient = np.dot(
            self.k.transpose(0, 2, 1),
            self._combine_heads(d_k)
        ).mean(axis=0)
        
        # Query weights gradient
        d_q = np.matmul(d_attention_weights, self.k) / np.sqrt(self.d_k)
        self.w_q_gradient = np.dot(
            self.q.transpose(0, 2, 1),
            self._combine_heads(d_q)
        ).mean(axis=0)
        
        # Return gradients for q, k, v
        return (
            np.dot(self._combine_heads(d_q), self.w_q.T),
            np.dot(self._combine_heads(d_k), self.w_k.T),
            np.dot(self._combine_heads(d_v), self.w_v.T)
        )

class TransformerBlock(Layer):
    """Transformer block with multi-head attention and feed-forward layers."""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        # Initialize sub-layers
        self.attention = MultiHeadAttention(num_heads, d_model, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        
        # Feed-forward weights
        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        
    def build(self, input_shape: tuple) -> None:
        """Initialize layer parameters."""
        # Initialize feed-forward weights
        limit = np.sqrt(6 / (self.d_model + self.d_ff))
        self.w1 = np.random.uniform(-limit, limit, (self.d_model, self.d_ff))
        self.w2 = np.random.uniform(-limit, limit, (self.d_ff, self.d_model))
        self.b1 = np.zeros(self.d_ff)
        self.b2 = np.zeros(self.d_model)
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass computation."""
        # Self-attention
        attention_output = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + attention_output)
        
        # Feed-forward network
        ff_output = np.dot(x, self.w1) + self.b1
        ff_output = self._activate(ff_output)
        if self.dropout > 0 and self.training:
            ff_output *= np.random.binomial(1, 1-self.dropout, ff_output.shape)
        ff_output = np.dot(ff_output, self.w2) + self.b2
        
        # Residual connection and layer normalization
        return self.layer_norm2(x + ff_output)
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation."""
        # Layer norm 2 backward
        d_ff_output = self.layer_norm2.backward(gradient)
        
        # Feed-forward backward
        d_ff = np.dot(d_ff_output, self.w2.T)
        self.w2_gradient = np.dot(
            self._activate(np.dot(self.norm1_output, self.w1) + self.b1).T,
            d_ff_output
        )
        self.b2_gradient = d_ff_output.sum(axis=(0, 1))
        
        if self.dropout > 0 and self.training:
            d_ff *= self.dropout_mask
            
        d_ff = self._activate_gradient(d_ff)
        self.w1_gradient = np.dot(self.norm1_output.T, d_ff)
        self.b1_gradient = d_ff.sum(axis=(0, 1))
        
        # Attention backward
        d_attention = self.attention.backward(self.layer_norm1.backward(d_ff))
        
        return d_attention
        
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        raise ValueError(f"Unsupported activation: {self.activation}")
        
    def _activate_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute activation function gradient."""
        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'gelu':
            # Approximate GELU gradient
            return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        raise ValueError(f"Unsupported activation: {self.activation}")

class LayerNorm(Layer):
    """Layer normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-12):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass computation."""
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.normalized = (x - self.mean) / self.std
        return self.gamma * self.normalized + self.beta
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation."""
        batch_size, seq_len, _ = gradient.shape
        
        # Gradients for gamma and beta
        self.gamma_gradient = np.sum(gradient * self.normalized, axis=(0, 1))
        self.beta_gradient = np.sum(gradient, axis=(0, 1))
        
        # Gradient for normalized input
        d_normalized = gradient * self.gamma
        
        # Gradient for variance
        d_var = -0.5 * np.sum(d_normalized * (self.inputs - self.mean) / 
                             (self.var + self.eps) ** 1.5, axis=-1, keepdims=True)
        
        # Gradient for mean
        d_mean = -np.sum(d_normalized / self.std, axis=-1, keepdims=True) - \
                 2 * d_var * np.mean(self.inputs - self.mean, axis=-1, keepdims=True)
        
        # Gradient for input
        return d_normalized / self.std + \
               2 * d_var * (self.inputs - self.mean) / (batch_size * seq_len) + \
               d_mean / (batch_size * seq_len) 