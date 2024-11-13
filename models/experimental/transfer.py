"""Transfer learning implementations."""

import numpy as np
from core.base import BaseEstimator

class TransferLearner(BaseEstimator):
    """Transfer learning implementation."""
    
    def __init__(self, base_model: BaseEstimator, target_model: BaseEstimator):
        super().__init__()
        self.base_model = base_model
        self.target_model = target_model
        
    def transfer_knowledge(self, X_source: np.ndarray, y_source: np.ndarray,
                         X_target: np.ndarray, y_target: np.ndarray):
        """Transfer knowledge from source to target domain."""
        self.base_model.fit(X_source, y_source)
        source_features = self.base_model.transform(X_target)
        self.target_model.fit(source_features, y_target)