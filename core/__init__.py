"""Core components of the ML library."""

from .base import Estimator, Optimizer, Loss, Layer, BaseTransformer
from .config import ModelConfig, TrainingConfig, ConfigManager
from .exceptions import (
    MLLibraryError, NotFittedError, ConvergenceError,
    ValidationError, DimensionalityError, ParameterError,
    ConfigurationError
)
from .validation import check_array, check_X_y, check_is_fitted
from .callbacks import Callback, EarlyStopping, ModelCheckpoint
from .logging import get_logger, TrainingLogger
from .metrics import (
    Metric, Accuracy, MSE, MAE,
    get_metric, MetricList
)
from .dtypes import (
    Number, Array, Features, Target,
    EPSILON, DEFAULT_RANDOM_STATE,
    SUPPORTED_OPTIMIZERS, SUPPORTED_ACTIVATIONS, SUPPORTED_LOSSES,
    DEFAULT_OPTIMIZER_CONFIG, DEFAULT_TRAINING_CONFIG
)

__all__ = [
    # Base classes
    'Estimator', 'Optimizer', 'Loss', 'Layer', 'BaseTransformer',
    
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
    
    # Types and Constants
    'Number', 'Array', 'Features', 'Target',
    'EPSILON', 'DEFAULT_RANDOM_STATE',
    'SUPPORTED_OPTIMIZERS', 'SUPPORTED_ACTIVATIONS', 'SUPPORTED_LOSSES',
    'DEFAULT_OPTIMIZER_CONFIG', 'DEFAULT_TRAINING_CONFIG'
]