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

from core import (
    Estimator, Optimizer, Loss, Layer,
    check_array, check_X_y, check_is_fitted,
    setup_logger, TrainingLogger
)

from models.linear import (
    ElasticNetRegression,
    AdamOptimizer,
    RMSpropOptimizer,
    ElasticNetLoss
)

from models.neural import (
    Layer,
    MLP,
    Activation,
    Dense
)

from models.svm import (
    SVM,
    AdvancedKernels,
    ActiveLearningStrategy,
    SparseSVM,
    RobustSVM
)

from models.spectral import (
    AdaptiveFourierModel,
    TimeFrequencyAnalyzer,
    ParametricSolver
)

from optimization.optimizers import (
    PSO,
    CMAESOptimizer,
    NelderMead,
    MultiObjectiveOptimizer
)

from utils import (
    ActiveLearner,
    TransferLearner,
    MetaLearner,
    EnsembleLearner,
    AutoML,
    ModelSelector,
    FeatureSelector
)

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
    
    # Types and Constants
    'Number', 'Array', 'Features', 'Target',
    'EPSILON', 'DEFAULT_RANDOM_STATE',
    'SUPPORTED_OPTIMIZERS', 'SUPPORTED_ACTIVATIONS', 'SUPPORTED_LOSSES',
    'DEFAULT_OPTIMIZER_CONFIG', 'DEFAULT_TRAINING_CONFIG'
]
