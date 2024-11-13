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
    ElasticNetRegression,
    ElasticNetLoss  # Added ElasticNet
)

from .svm import (
    SVMModelSelector,
    SVM,
    AdvancedKernels,
    SparseSVM,
    RobustSVM,
    IncrementalSVM,
    UncertaintySampling,
    DiversitySampling

)

from .trees import (
    DecisionTreeClassifier,
    RandomForestClassifier,
)

from .ensemble import (
    BaseBooster,
    GradientBoostingRegressor,
    XGBoostRegressor,
    LightGBMRegressor

)

from .neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    AdaptiveKNN,
    LocallyWeightedRegressor,
    VoronoiClassifier,
    PrototypeSelector
)

from .topic import (
    LatentDirichletAllocation,
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

from .spectral import (
    BaseFourierModel,
    AdaptiveFourierModel,
    SpectralDESolver,
    SpectralKernel,
    TimeFrequencyAnalyzer
)

# Explicitly list exports from each submodule
__all__ = [
    # Neural networks
    'NeuralNetwork',
    'CNN',
    'RNN',
    'LSTM',
    'Transformer',
    
    # Linear models
    'LinearRegression',
    'LogisticRegression',
    'RidgeRegression',
    'LassoRegression',
    
    # SVM models
    'SVM',
    'AdvancedKernels', 
    'SparseSVM',
    'RobustSVM',
    'IncrementalSVM',
    'UncertaintySampling',
    'DiversitySampling',
    'SVMModelSelector',
    
    # Tree models
    'DecisionTreeClassifier',
    'RandomForestClassifier', 

    
    # Ensemble models
    'BaseBooster',
    'GradientBoostingRegressor',
    'XGBoostRegressor',
    'LightGBMRegressor',

    
    # Neighbors
    'KNeighborsClassifier',
    'KNeighborsRegressor', 
    'AdaptiveKNN',
    'LocallyWeightedRegressor',
    'VoronoiClassifier',
    'PrototypeSelector',

    
    # Topic models
    'LatentDirichletAllocation',

    # Spectral models
    'BaseFourierModel',
    'AdaptiveFourierModel',
    'SpectralDESolver',
    'SpectralKernel',
    'TimeFrequencyAnalyzer',
    
    # Experimental models
    'QuantumNeuralNetwork',
    'CapsuleNetwork',
    'GraphNeuralNetwork',
    'AutoML',
    
    # Classification
    'LDA',
    'BaseClassifier',
    'SoftmaxClassifier', 
    'SVMClassifier',
    'EnsembleClassifier'
]