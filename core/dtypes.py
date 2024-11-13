"""Common data types and constants."""

from typing import TypeVar, Union, Dict, List
import numpy as np

# Type variables
Number = Union[int, float, np.number]
Array = TypeVar('Array', np.ndarray, List[float], List[int])
Features = TypeVar('Features', np.ndarray, List[List[float]])
Target = TypeVar('Target', np.ndarray, List[float], List[int])

# Constants
EPSILON = 1e-7
DEFAULT_RANDOM_STATE = 42
SUPPORTED_OPTIMIZERS = ['sgd', 'adam', 'rmsprop']
SUPPORTED_ACTIVATIONS = ['relu', 'sigmoid', 'tanh']
SUPPORTED_LOSSES = ['mse', 'binary_crossentropy', 'categorical_crossentropy']

# Default configurations
DEFAULT_OPTIMIZER_CONFIG = {
    'learning_rate': 0.01,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': EPSILON
}

DEFAULT_TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'shuffle': True
} 