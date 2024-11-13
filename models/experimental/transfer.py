"""Transfer learning implementations."""

import numpy as np
from core import (
    Estimator,
    check_array,
    check_X_y,
    check_is_fitted,
    EPSILON,
    DEFAULT_RANDOM_STATE
)

class TransferLearner(Estimator):
    """Transfer learning implementation."""
    
    def __init__(self, base_model: Estimator, target_model: Estimator):
        """Initialize transfer learner.
        
        Args:
            base_model: Model trained on source domain
            target_model: Model to be trained on target domain
        """
        super().__init__()
        if not hasattr(base_model, 'transform'):
            raise ValueError("Base model must implement transform() method")
            
        self.base_model = base_model
        self.target_model = target_model
        self.is_fitted_ = False
        
    def transfer_knowledge(self, X_source: np.ndarray, y_source: np.ndarray,
                         X_target: np.ndarray, y_target: np.ndarray) -> 'TransferLearner':
        """Transfer knowledge from source to target domain.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels  
            X_target: Target domain features
            y_target: Target domain labels
            
        Returns:
            self: Fitted estimator
        """
        # Input validation
        X_source, y_source = check_X_y(X_source, y_source)
        X_target, y_target = check_X_y(X_target, y_target)
        
        if X_source.shape[1] != X_target.shape[1]:
            raise ValueError("Source and target domains must have same number of features")
            
        # Fit base model on source domain
        self.base_model.fit(X_source, y_source)
        
        # Transform target data using base model
        try:
            source_features = self.base_model.transform(X_target)
        except Exception as e:
            raise RuntimeError(f"Error transforming target data: {str(e)}")
            
        # Fit target model on transformed features
        self.target_model.fit(source_features, y_target)
        self.is_fitted_ = True
        
        return self