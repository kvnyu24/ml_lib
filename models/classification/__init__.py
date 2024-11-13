"""Classification model implementations."""

from .base import BaseClassifier
from .classifiers import (
    SoftmaxClassifier,
    SVMClassifier,
    EnsembleClassifier
)
from .lda import LDA

__all__ = [
    'BaseClassifier',
    'SoftmaxClassifier',
    'SVMClassifier',
    'EnsembleClassifier',
    'LDA'
] 