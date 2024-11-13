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

__version__ = '0.1.0'
