"""Transfer learning implementations with robust domain adaptation."""

import numpy as np
from typing import Optional, Union, Dict
from core import (
    Estimator, 
    check_array,
    check_X_y,
    check_is_fitted,
    EPSILON,
    DEFAULT_RANDOM_STATE,
    ValidationError
)

class TransferLearner(Estimator):
    """Transfer learning implementation with multiple adaptation strategies."""
    
    ADAPTATION_STRATEGIES = ['fine_tune', 'feature_extraction', 'domain_adaptation']
    
    def __init__(
        self,
        base_model: Estimator,
        target_model: Estimator,
        adaptation_strategy: str = 'fine_tune',
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        random_state: Optional[int] = None
    ):
        """Initialize transfer learner.
        
        Args:
            base_model: Model trained on source domain
            target_model: Model to be trained on target domain
            adaptation_strategy: Strategy for knowledge transfer
            learning_rate: Learning rate for fine-tuning
            n_epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Epochs to wait before early stopping
            random_state: Random seed for reproducibility
        """
        super().__init__()
        if not hasattr(base_model, 'transform'):
            raise ValidationError("Base model must implement transform() method")
            
        if adaptation_strategy not in self.ADAPTATION_STRATEGIES:
            raise ValidationError(f"Strategy must be one of {self.ADAPTATION_STRATEGIES}")
            
        self.base_model = base_model
        self.target_model = target_model
        self.adaptation_strategy = adaptation_strategy
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state or DEFAULT_RANDOM_STATE
        self.is_fitted_ = False
        
    def _domain_adaptation(self, source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
        """Align source and target feature distributions."""
        # Compute domain means
        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)
        
        # Center the distributions
        source_centered = source_features - source_mean
        target_centered = target_features - target_mean
        
        # Compute correlation matrices
        source_corr = np.dot(source_centered.T, source_centered) / (len(source_features) - 1)
        target_corr = np.dot(target_centered.T, target_centered) / (len(target_features) - 1)
        
        # Whitening transform
        eps = 1e-6
        source_whitened = np.dot(source_centered, np.linalg.inv(np.sqrt(source_corr + eps * np.eye(source_corr.shape[0]))))
        target_whitened = np.dot(target_centered, np.linalg.inv(np.sqrt(target_corr + eps * np.eye(target_corr.shape[0]))))
        
        return target_whitened
        
    def transfer_knowledge(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray
    ) -> 'TransferLearner':
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
            raise ValidationError("Source and target domains must have same number of features")
            
        # Set random seed
        np.random.seed(self.random_state)
        
        # Fit base model on source domain
        self.base_model.fit(X_source, y_source)
        
        # Transform data using base model
        try:
            source_features = self.base_model.transform(X_source)
            target_features = self.base_model.transform(X_target)
        except Exception as e:
            raise RuntimeError(f"Error transforming data: {str(e)}")
            
        # Apply adaptation strategy
        if self.adaptation_strategy == 'domain_adaptation':
            target_features = self._domain_adaptation(source_features, target_features)
            
        # Split target data for validation
        n_val = int(len(target_features) * self.validation_split)
        indices = np.random.permutation(len(target_features))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train = target_features[train_idx]
        y_train = y_target[train_idx]
        X_val = target_features[val_idx]
        y_val = y_target[val_idx]
        
        # Train target model with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.n_epochs):
            # Mini-batch training
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i + self.batch_size]
                batch_y = y_train[i:i + self.batch_size]
                self.target_model.partial_fit(batch_X, batch_y, self.learning_rate)
            
            # Validation
            val_loss = self.target_model.compute_loss({'X': X_val, 'y': y_val})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.target_model.get_params()
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                break
                
        # Restore best model
        if best_model_state is not None:
            self.target_model.set_params(**best_model_state)
            
        self.is_fitted_ = True
        return self