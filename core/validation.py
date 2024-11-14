"""Input validation utilities."""

import numpy as np
from typing import Optional, Tuple, Union, List
from .exceptions import ValidationError
from .dtypes import Features, Target

def check_array(X: Features,
                allow_nd: bool = False,
                ensure_2d: bool = True,
                dtype: Optional[Union[str, np.dtype]] = None) -> np.ndarray:
    """Validate array input."""
    try:
        X = np.asarray(X)
    except Exception as e:
        raise ValidationError(f"Error converting input to numpy array: {e}")
        
    if ensure_2d:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            if not allow_nd:
                raise ValidationError(
                    f"Expected 2D array, got {X.ndim}D array instead"
                )
                
    if dtype is not None:
        try:
            X = X.astype(dtype)
        except Exception as e:
            raise ValidationError(f"Error converting array to dtype {dtype}: {e}")
            
    return X

def check_X_y(X: Features,
              y: Target,
              multi_output: bool = False,
              allow_nd: bool = False,
              dtype: Optional[Union[str, np.dtype]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Validate feature and target arrays."""
    X = check_array(X, allow_nd=allow_nd, dtype=dtype)
    y = check_array(y, allow_nd=multi_output, ensure_2d=False, dtype=dtype)
    
    if X.shape[0] != y.shape[0]:
        raise ValidationError(
            f"X and y have incompatible shapes: {X.shape} vs {y.shape}"
        )
        
    return X, y

def check_is_fitted(estimator: object,
                   attributes: Optional[Union[str, List[str]]] = None,
                   skip_validation: bool = False) -> None:
    """Check if estimator is fitted."""
    # Skip validation if explicitly requested
    if skip_validation:
        return
        
    # Check if either is_fitted attribute exists and is True, or if specified attributes exist
    if attributes:
        if isinstance(attributes, str):
            attributes = [attributes]
        for attr in attributes:
            if not hasattr(estimator, attr):
                raise ValidationError(
                    f"Attribute {attr} not found. {estimator.__class__.__name__} "
                    "instance is not fitted correctly"
                )
    else:
        # If no specific attributes are provided, check for common fitted attributes
        fitted_attrs = [v for v in vars(estimator) if v.endswith('_') and not v.startswith('__')]
        if not fitted_attrs:
            raise ValidationError(
                f"{estimator.__class__.__name__} is not fitted yet. "
                "Call 'fit' before using this estimator."
            )