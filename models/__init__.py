"""Models Package

This package contains various machine learning models including:
- Neural network architectures (MLP, CNN, ResNet)
- Linear models (LinearRegression, LogisticRegression) 
- Support vector machines (SVM, SVR)
- Decision trees and random forests (DecisionTree, RandomForest)
- Ensemble methods (Bagging, Boosting)
"""

from .neural import (
    MLP,
    Conv2D,
    ResidualBlock,
    BatchNormalization,
    GlobalAveragePooling2D, 
    MaxPool2D,
    Dense,
    Dropout,
    Flatten,
    ResNet  # Added ResNet
)

from .linear import (
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNet  # Added ElasticNet
)

from .svm import (
    SVM,
    SVR,
    LinearSVM,
    KernelSVM  # Added KernelSVM
)

from .trees import (
    DecisionTree,
    RandomForest,
    ExtraTreesClassifier,
    GradientBoostedTrees  # Added GradientBoostedTrees
)

from .ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    AdaBoost,
    GradientBoosting,
    StackingClassifier,  # Added StackingClassifier
    VotingClassifier  # Added VotingClassifier
)

from .neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    NearestCentroid
)

from .topic import (
    LatentDirichletAllocation,
    NMF,
    LSA,
    PLSA
)

from .experimental import (
    QuantumNeuralNetwork,
    CapsuleNetwork,
    GraphNeuralNetwork,
    AutoML
)

from .classification import (
    LDA,
    BaseClassifier,
    SoftmaxClassifier,
    SVMClassifier,
    EnsembleClassifier,

)

__all__ = [
    # Neural Networks
    'MLP',
    'Conv2D', 
    'ResidualBlock',
    'BatchNormalization',
    'GlobalAveragePooling2D',
    'MaxPool2D',
    'Dense',
    'Dropout',
    'Flatten',
    'ResNet',  # Added ResNet

    # Linear Models
    'LinearRegression',
    'LogisticRegression',
    'RidgeRegression', 
    'LassoRegression',
    'ElasticNet',  # Added ElasticNet

    # Support Vector Machines
    'SVM',
    'SVR', 
    'LinearSVM',
    'KernelSVM',  # Added KernelSVM

    # Decision Trees
    'DecisionTree',
    'RandomForest',
    'ExtraTreesClassifier',
    'GradientBoostedTrees',  # Added GradientBoostedTrees

    # Ensemble Methods
    'BaggingClassifier',
    'BaggingRegressor',
    'AdaBoost',
    'GradientBoosting',
    'StackingClassifier',  # Added StackingClassifier
    'VotingClassifier',  # Added VotingClassifier

    # Neighbors
    'KNeighborsClassifier',
    'KNeighborsRegressor', 
    'RadiusNeighborsClassifier',
    'NearestCentroid',

    # Topic Models
    'LatentDirichletAllocation',
    'NMF',
    'LSA',
    'PLSA',

    # Experimental Models
    'QuantumNeuralNetwork',
    'CapsuleNetwork',
    'GraphNeuralNetwork',
    'AutoML',

    # Classification Models
    'LDA',
    'BaseClassifier',
    'SoftmaxClassifier',
    'SVMClassifier',
    'EnsembleClassifier',
]