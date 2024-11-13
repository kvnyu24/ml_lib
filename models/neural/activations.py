"""Activation function implementations."""

import numpy as np
from typing import Callable
from core import EPSILON

class Activation:
    """Base class for activation functions."""
    
    @staticmethod
    def get(name: str) -> tuple[Callable, Callable]:
        """Get activation function and its derivative."""
        if name == 'relu':
            return Activation.relu, Activation.relu_prime
        elif name == 'sigmoid':
            return Activation.sigmoid, Activation.sigmoid_prime
        elif name == 'tanh':
            return Activation.tanh, Activation.tanh_prime
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
        
    @staticmethod
    def relu_prime(x: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return np.where(x > 0, 1, 0)
        
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    @staticmethod
    def sigmoid_prime(x: np.ndarray) -> np.ndarray:
        """Sigmoid derivative."""
        s = Activation.sigmoid(x)
        return s * (1 - s)
        
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
        
    @staticmethod
    def tanh_prime(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent derivative."""
        return 1 - np.tanh(x) ** 2 