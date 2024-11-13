"""Meta-learning and few-shot learning implementations."""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from core import (
    Estimator, 
    ValidationError,
    EPSILON,
    DEFAULT_RANDOM_STATE,
    check_array
)

class MetaLearner:
    """Model-agnostic meta-learning (MAML) implementation.
    
    Implements the MAML algorithm for few-shot learning by adapting a base model
    to new tasks through gradient descent with support for:
    - Early stopping based on validation performance
    - Gradient clipping to prevent exploding gradients
    - Learning rate scheduling
    - Multiple adaptation strategies
    """
    
    ADAPTATION_STRATEGIES = ['gradient_descent', 'momentum', 'adam']
    
    def __init__(self,
                 base_model: Estimator,
                 n_inner_steps: int = 5,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 adaptation_strategy: str = 'adam',
                 clip_value: float = 5.0,
                 early_stopping_patience: int = 5,
                 validation_split: float = 0.2,
                 lr_schedule_factor: float = 0.5,
                 lr_schedule_patience: int = 3,
                 random_state: Optional[int] = None):
        """Initialize the meta-learner.
        
        Args:
            base_model: Base model to adapt
            n_inner_steps: Number of inner loop optimization steps
            inner_lr: Learning rate for inner loop optimization
            outer_lr: Learning rate for outer loop optimization
            adaptation_strategy: Strategy for parameter updates
            clip_value: Maximum allowed gradient norm
            early_stopping_patience: Epochs to wait before early stopping
            validation_split: Fraction of support set to use for validation
            lr_schedule_factor: Factor to reduce learning rate by
            lr_schedule_patience: Epochs to wait before reducing learning rate
            random_state: Random seed for reproducibility
        """
        if not isinstance(base_model, Estimator):
            raise ValidationError("base_model must be an instance of Estimator")
            
        if n_inner_steps < 1:
            raise ValidationError("n_inner_steps must be greater than 0")
            
        if not 0 < inner_lr <= 1 or not 0 < outer_lr <= 1:
            raise ValidationError("Learning rates must be between 0 and 1")
            
        if adaptation_strategy not in self.ADAPTATION_STRATEGIES:
            raise ValidationError(f"adaptation_strategy must be one of {self.ADAPTATION_STRATEGIES}")
            
        if not 0 < validation_split < 1:
            raise ValidationError("validation_split must be between 0 and 1")
            
        self.base_model = base_model
        self.n_inner_steps = n_inner_steps
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.adaptation_strategy = adaptation_strategy
        self.clip_value = clip_value
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.lr_schedule_factor = lr_schedule_factor
        self.lr_schedule_patience = lr_schedule_patience
        self.random_state = random_state or DEFAULT_RANDOM_STATE
        self.rng = np.random.RandomState(self.random_state)
        
        # Initialize optimizer state
        self.momentum = None
        self.adam_m = None
        self.adam_v = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        
    def _split_support_set(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split support set into train and validation sets."""
        n_samples = len(X)
        n_val = int(n_samples * self.validation_split)
        indices = self.rng.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
        
    def _update_params(self, model: Estimator, grads: Dict, lr: float) -> None:
        """Update model parameters using specified adaptation strategy."""
        if self.adaptation_strategy == 'gradient_descent':
            model.update_params(grads, lr)
            
        elif self.adaptation_strategy == 'momentum':
            if self.momentum is None:
                self.momentum = {k: np.zeros_like(v) for k, v in grads.items()}
            for k in grads:
                self.momentum[k] = self.beta1 * self.momentum[k] + (1 - self.beta1) * grads[k]
                model.update_params({k: self.momentum[k]}, lr)
                
        elif self.adaptation_strategy == 'adam':
            if self.adam_m is None or self.adam_v is None:
                self.adam_m = {k: np.zeros_like(v) for k, v in grads.items()}
                self.adam_v = {k: np.zeros_like(v) for k, v in grads.items()}
            
            for k in grads:
                self.adam_m[k] = self.beta1 * self.adam_m[k] + (1 - self.beta1) * grads[k]
                self.adam_v[k] = self.beta2 * self.adam_v[k] + (1 - self.beta2) * np.square(grads[k])
                m_hat = self.adam_m[k] / (1 - self.beta1)
                v_hat = self.adam_v[k] / (1 - self.beta2)
                model.update_params({k: m_hat / (np.sqrt(v_hat) + EPSILON)}, lr)
        
    def adapt(self, support_set: Dict[str, np.ndarray]) -> Estimator:
        """Adapt model to new task using support set with robust optimization."""
        if not isinstance(support_set, dict) or 'X' not in support_set or 'y' not in support_set:
            raise ValidationError("support_set must be dict with 'X' and 'y' keys")
            
        X = check_array(support_set['X'])
        y = check_array(support_set['y'])
        
        if len(X) != len(y):
            raise ValidationError("X and y must have same first dimension")
            
        if len(X) == 0:
            raise ValidationError("support_set cannot be empty")
            
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = self._split_support_set(X, y)
        
        adapted_model = self.base_model.clone()
        best_model = None
        best_val_loss = float('inf')
        patience_counter = 0
        lr_patience_counter = 0
        current_lr = self.inner_lr
        
        try:
            for step in range(self.n_inner_steps):
                # Compute and clip gradients
                grads = adapted_model.compute_gradients({'X': X_train, 'y': y_train})
                if not grads:
                    raise ValidationError("Model returned invalid gradients")
                    
                grad_norm = np.sqrt(sum(np.sum(np.square(g)) for g in grads.values()))
                if grad_norm > self.clip_value:
                    scale = self.clip_value / grad_norm
                    grads = {k: v * scale for k, v in grads.items()}
                
                # Update model parameters
                self._update_params(adapted_model, grads, current_lr)
                
                # Evaluate on validation set
                val_loss = adapted_model.compute_loss({'X': X_val, 'y': y_val})
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = adapted_model.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    lr_patience_counter += 1
                
                # Learning rate scheduling
                if lr_patience_counter >= self.lr_schedule_patience:
                    current_lr *= self.lr_schedule_factor
                    lr_patience_counter = 0
                
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    break
                    
        except Exception as e:
            raise ValidationError(f"Error during model adaptation: {str(e)}")
            
        return best_model if best_model is not None else adapted_model