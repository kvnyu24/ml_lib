"""Base optimizer implementations."""

from typing import Dict, Optional
from core import Optimizer, EPSILON
import numpy as np

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate=learning_rate)
        
    def compute_update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return params - self.learning_rate * gradients

class Adam(Optimizer):
    """Adam optimizer with momentum and adaptive learning rates."""
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = EPSILON):
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._state = {}  # Initialize empty state dict
        
    def initialize(self, param_shape: tuple) -> None:
        """Initialize optimizer state with the right shapes."""
        self._state['m'] = np.zeros(param_shape)  # First moment
        self._state['v'] = np.zeros(param_shape)  # Second moment
        self._iteration = 1  # Start from 1 to avoid division by zero
        
    def compute_update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if 'm' not in self._state:
            self.initialize(params.shape)
            
        m = self._state['m']
        v = self._state['v']
        
        m = self.beta1 * m + (1 - self.beta1) * gradients
        v = self.beta2 * v + (1 - self.beta2) * gradients**2
        
        m_hat = m / (1 - self.beta1**self._iteration)
        v_hat = v / (1 - self.beta2**self._iteration)
        
        self._state['m'] = m
        self._state['v'] = v
        self._iteration += 1
        
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)