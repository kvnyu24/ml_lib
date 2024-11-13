"""
Neural Network Library
=====================

This library implements a modular neural network framework with the following components:

- Layer classes for building network architectures
- Loss functions for training
- MLP class for combining layers into a network
- Utility functions for array operations and visualization
- Optimizers for gradient descent variants
- Regularization options
- Model saving/loading
- Training callbacks

The implementation follows modern deep learning practices with clean interfaces
for building and training neural networks.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Tuple, Callable
import pickle
import json
from pathlib import Path

from core import (
    Layer as BaseLayer,
    Loss as BaseLoss,
    Estimator,
    check_array,
    check_X_y,
    check_is_fitted,
    EPSILON,
    DEFAULT_RANDOM_STATE,
    SUPPORTED_ACTIVATIONS,
    Number,
    Array,
    Features,
    Target,
    get_logger
)

logger = get_logger(__name__)