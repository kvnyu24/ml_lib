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

# Advanced learning models
from .meta import MetaLearner
from .transfer import TransferLearner
from .ensemble import EnsembleLearner
from .automl import AutoML

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
    
    # Evaluation
    'ModelEvaluator',
    'ModelSelector'
] 