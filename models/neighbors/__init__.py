"""K-Nearest Neighbors algorithms."""

from .knn import KNeighborsClassifier, KNeighborsRegressor
from .advanced_nn import AdaptiveKNN, LocallyWeightedRegressor, VoronoiClassifier, PrototypeSelector

__all__ = ['KNeighborsClassifier', 'KNeighborsRegressor', 'AdaptiveKNN', 'LocallyWeightedRegressor', 'VoronoiClassifier', 'PrototypeSelector'] 