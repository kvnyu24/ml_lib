"""Preprocessing pipeline utilities."""

from typing import List, Union, Optional
from core import Transformer
import numpy as np

class PreprocessingPipeline:
    """Pipeline for combining multiple preprocessing steps."""
    
    def __init__(self, steps: List[Union[tuple, Transformer]]):
        self.steps = []
        for step in steps:
            if isinstance(step, tuple):
                name, transformer = step
            else:
                name = type(step).__name__.lower()
                transformer = step
            self.steps.append((name, transformer))
            
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PreprocessingPipeline':
        """Fit all transformers in the pipeline."""
        X_transformed = X
        for name, transformer in self.steps:
            X_transformed = transformer.fit_transform(X_transformed, y)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all transformations to X."""
        X_transformed = X
        for name, transformer in self.steps:
            X_transformed = transformer.transform(X_transformed)
        return X_transformed
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform data in one step."""
        return self.fit(X, y).transform(X) 