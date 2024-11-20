"""Feature engineering utilities."""

import numpy as np
from typing import Optional, List, Union, Callable, Tuple
from itertools import combinations, combinations_with_replacement
from core import (
    Transformer,
    check_array,
    get_logger,
    ValidationError
)

logger = get_logger(__name__)

class PolynomialFeatures(Transformer):
    """Generate polynomial and interaction features."""
    
    def __init__(self, degree: int = 2, interaction_only: bool = False):
        if not isinstance(degree, int) or degree < 0:
            raise ValidationError("degree must be a non-negative integer")
        self.degree = degree
        self.interaction_only = interaction_only
        self.n_output_features_: Optional[int] = None
        self.powers_: Optional[List[np.ndarray]] = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PolynomialFeatures':
        """Compute number of output features and powers."""
        X = check_array(X)
        n_features = X.shape[1]
        
        combinations_list = []
        for d in range(0, self.degree + 1):
            if self.interaction_only and d > 1:
                combinations_list.extend(combinations(range(n_features), d))
            else:
                combinations_list.extend(combinations_with_replacement(range(n_features), d))
        
        self.powers_ = []
        for combo in combinations_list:
            power = np.zeros(n_features, dtype=int)
            for i in combo:
                power[i] += 1
            self.powers_.append(power)
                
        self.n_output_features_ = len(self.powers_)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to polynomial features."""
        X = check_array(X)
        n_samples = X.shape[0]
        
        if self.powers_ is None or self.n_output_features_ is None:
            raise ValidationError("PolynomialFeatures must be fitted before transform")
            
        X_new = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i, power in enumerate(self.powers_):
            X_new[:, i] = np.prod(X ** power, axis=1)
            
        return X_new

class InteractionFeatures(Transformer):
    """Generate interaction features between specified columns."""
    
    def __init__(self, interaction_pairs: Optional[List[Tuple[int, int]]] = None, 
                 interaction_type: str = 'multiply'):
        """
        Args:
            interaction_pairs: List of tuples containing column indices to interact
            interaction_type: Type of interaction - 'multiply', 'add', 'subtract', or 'divide'
        """
        valid_types = {'multiply', 'add', 'subtract', 'divide'}
        if interaction_type not in valid_types:
            raise ValidationError(f"interaction_type must be one of: {', '.join(valid_types)}")

        self.interaction_pairs = interaction_pairs
        self.interaction_type = interaction_type
        self._interaction_func = None
        self._set_interaction_func()
        
    def _set_interaction_func(self):
        """Set the interaction function based on interaction type."""
        interaction_funcs = {
            'multiply': np.multiply,
            'add': np.add,
            'subtract': np.subtract,
            'divide': np.divide
        }
        if self.interaction_type not in interaction_funcs:
            raise ValidationError("interaction_type must be one of: multiply, add, subtract, divide")
        self._interaction_func = interaction_funcs[self.interaction_type]
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'InteractionFeatures':
        """Fit transformer."""
        X = check_array(X)
        n_features = X.shape[1]
        
        if self.interaction_pairs is None:
            self.interaction_pairs = list(combinations(range(n_features), 2))
            
        # Validate pairs
        for pair in self.interaction_pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValidationError("interaction_pairs must be tuples of length 2")
            if not all(0 <= idx < n_features for idx in pair):
                raise ValidationError(f"Invalid column indices in pair {pair}")
                
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Generate interaction features."""
        X = check_array(X)
        n_samples = X.shape[0]
        
        if self.interaction_pairs is None:
            raise ValidationError("InteractionFeatures must be fitted before transform")
        
        # Original features plus interaction features
        n_interactions = len(self.interaction_pairs)
        X_new = np.empty((n_samples, X.shape[1] + n_interactions))
        X_new[:, :X.shape[1]] = X
        
        # Add error checking for division by zero and overflow
        for i, (col1, col2) in enumerate(self.interaction_pairs):
            try:
                if self.interaction_type == 'divide':
                    # Handle division by zero
                    denominator = X[:, col2]
                    # Replace zeros with small epsilon to avoid division by zero
                    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
                    X_new[:, X.shape[1] + i] = X[:, col1] / denominator
                else:
                    X_new[:, X.shape[1] + i] = self._interaction_func(X[:, col1], X[:, col2])
                
                # Check for overflow/underflow
                if np.any(~np.isfinite(X_new[:, X.shape[1] + i])):
                    logger.warning(f"Interaction between columns {col1} and {col2} produced infinite/NaN values")
                    # Replace inf/nan with max/min finite values
                    X_new[:, X.shape[1] + i] = np.nan_to_num(X_new[:, X.shape[1] + i])
                    
            except Exception as e:
                logger.error(f"Error computing interaction between columns {col1} and {col2}: {str(e)}")
                raise
                
        return X_new

class CustomFeatureTransformer(Transformer):
    """Apply custom transformations to features."""
    
    def __init__(self, transformations: List[Union[str, Callable]], 
                 columns: Optional[List[int]] = None):
        """
        Args:
            transformations: List of transformations to apply. Can be string names
                           of common transforms or custom callable functions
            columns: List of column indices to transform. If None, applies to all columns
        """
        self.transformations = transformations
        self.columns = columns
        self._transform_funcs: List[Callable] = []
        
        # Map common transformation names to functions
        transform_map = {
            'log': np.log,
            'sqrt': np.sqrt,
            'square': np.square,
            'abs': np.abs,
            'reciprocal': np.reciprocal
        }
        
        for t in transformations:
            if isinstance(t, str):
                func = transform_map.get(t.lower())
                if func is None:
                    raise ValidationError(f"Unknown transformation: {t}")
                self._transform_funcs.append(func)
            elif callable(t):
                self._transform_funcs.append(t)
            else:
                raise ValidationError("Transformations must be strings or callable functions")
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CustomFeatureTransformer':
        """Fit transformer."""
        X = check_array(X)
        if self.columns is None:
            self.columns = list(range(X.shape[1]))
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply custom transformations."""
        X = check_array(X)
        X_new = X.copy()
        
        if self.columns is None:
            raise ValidationError("CustomFeatureTransformer must be fitted before transform")
            
        for col in self.columns:
            for func in self._transform_funcs:
                try:
                    X_new[:, col] = func(X_new[:, col])
                except Exception as e:
                    logger.error(f"Error applying transformation to column {col}: {str(e)}")
                    raise
                    
        return X_new