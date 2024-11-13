"""Gradient-based optimization algorithms."""

import numpy as np
from typing import Dict, Optional
from core import Optimizer, EPSILON

class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self,
                 learning_rate: float = 0.001,
                 rho: float = 0.9,
                 epsilon: float = EPSILON):
        super().__init__(learning_rate=learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self._state['square_avg'] = None
        
    def compute_update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if 'square_avg' not in self._state:
            self._state['square_avg'] = np.zeros_like(params)
            
        square_avg = self._state['square_avg']
        square_avg = self.rho * square_avg + (1 - self.rho) * gradients**2
        self._state['square_avg'] = square_avg
        
        return params - self.learning_rate * gradients / (np.sqrt(square_avg) + self.epsilon)

class Momentum(Optimizer):
    """Momentum optimizer."""
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9):
        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self._state['velocity'] = None
        
    def compute_update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if 'velocity' not in self._state:
            self._state['velocity'] = np.zeros_like(params)
            
        velocity = self._state['velocity']
        velocity = self.momentum * velocity - self.learning_rate * gradients
        self._state['velocity'] = velocity
        
        return params + velocity

class Adagrad(Optimizer):
    """Adagrad optimizer."""
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 epsilon: float = EPSILON):
        super().__init__(learning_rate=learning_rate)
        self.epsilon = epsilon
        self._state['sum_squares'] = None
        
    def compute_update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if 'sum_squares' not in self._state:
            self._state['sum_squares'] = np.zeros_like(params)
            
        sum_squares = self._state['sum_squares']
        sum_squares += gradients**2
        self._state['sum_squares'] = sum_squares
        
        return params - self.learning_rate * gradients / (np.sqrt(sum_squares) + self.epsilon) 