"""Model implementations and utilities."""

from .linear import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression
)

from .neural import (
    MLP,
    Layer,
    Dense,
    Activation
)

from .svm import (
    SVM,
    KernelSVM,
    LinearSVM
)

from .evaluation import (
    ModelEvaluator,
    ModelSelector,
    CrossValidationSplitter
)

from .feature_selection import (
    FeatureSelector,
    VarianceThreshold,
    MutualInformation
)

from .online import (
    OnlineLearner,
    StreamingModel
)

__all__ = [
    # Linear models
    'LinearRegression',
    'RidgeRegression',
    'LassoRegression',
    'ElasticNetRegression',
    
    # Neural networks
    'MLP',
    'Layer',
    'Dense',
    'Activation',
    
    # SVM models
    'SVM',
    'KernelSVM',
    'LinearSVM',
    
    # Model evaluation
    'ModelEvaluator',
    'ModelSelector',
    'CrossValidationSplitter',
    
    # Feature selection
    'FeatureSelector',
    'VarianceThreshold',
    'MutualInformation',
    
    # Online learning
    'OnlineLearner',
    'StreamingModel'
]