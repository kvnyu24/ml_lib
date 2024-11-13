"""Machine Learning Algorithm Library

A comprehensive library for:
- Spectral analysis and Fourier methods
- Classification and model selection
- Geometric distance calculations
- Optimization algorithms
- Neural networks and deep learning
- Support vector machines
- Linear models
"""
from core import *
from models import *
from optimization import *
from utils import *

__version__ = '0.1.0'

__all__ = [
    # Base classes
    'Estimator', 'Optimizer', 'Loss', 'Layer', 'Transformer',
    
    # Configuration
    'ModelConfig', 'TrainingConfig', 'ConfigManager',
    
    # Exceptions
    'MLLibraryError', 'NotFittedError', 'ConvergenceError',
    'ValidationError', 'DimensionalityError', 'ParameterError',
    'ConfigurationError',
    
    # Validation
    'check_array', 'check_X_y', 'check_is_fitted',
    
    # Callbacks
    'Callback', 'EarlyStopping', 'ModelCheckpoint',
    
    # Logging
    'get_logger', 'TrainingLogger',
    
    # Metrics
    'Metric', 'Accuracy', 'MSE', 'MAE',
    'get_metric', 'MetricList',
    
    # Neural Network Layers
    'Conv2D', 'BatchNormalization', 'GlobalAveragePooling2D',
    'MaxPool2D', 'Dropout', 'Flatten', 'Dense', 'ResidualBlock',
    
    # Types and Constants
    'Number', 'Array', 'Features', 'Target',
    'EPSILON', 'DEFAULT_RANDOM_STATE',
    'SUPPORTED_OPTIMIZERS', 'SUPPORTED_ACTIVATIONS', 'SUPPORTED_LOSSES',
    'DEFAULT_OPTIMIZER_CONFIG', 'DEFAULT_TRAINING_CONFIG'
]
