"""Model implementations."""

# Classification models
from .classification import (
    BaseClassifier,
    SoftmaxClassifier,
    SVMClassifier,
    EnsembleClassifier
)

# Tree-based models
from .trees import (
    DecisionTreeClassifier,
    RandomForestClassifier
)

# SVM models
from .svm import (
    IncrementalSVM,
    SVMModelSelector
)

from .linear import (
    ElasticNetRegression,
    ElasticNetLoss
)


# Advanced learning models
from .experimental import (
    ActiveLearner,
    MetaLearner,
    TransferLearner,
    AutoML,
    NeuralODE
)

from .topic import (
    LatentDirichletAllocation
)

from .ensemble import EnsembleLearner

# Evaluation utilities
from .evaluation import (
    ModelEvaluator,
    ModelSelector
)

__all__ = [
    # Classification
    'BaseClassifier',
    'SoftmaxClassifier',
    'SVMClassifier',
    'EnsembleClassifier',

    # Linear models
    'ElasticNetRegression',
    'ElasticNetLoss',
    
    # Trees
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    
    # SVM
    'IncrementalSVM',
    'SVMModelSelector',
    
    # Advanced Learning
    'MetaLearner',
    'TransferLearner',
    'EnsembleLearner',
    'AutoML',
    'NeuralODE',

    # Topic modeling
    'LatentDirichletAllocation',

    # Evaluation
    'ModelEvaluator',
    'ModelSelector'
] 