"""Active learning implementations."""

import numpy as np
from typing import Optional, List, Union, Dict
from scipy.stats import entropy
from core import (
    Estimator,
    DEFAULT_RANDOM_STATE,
    ValidationError,
    check_array,
    euclidean_distance
)
from models.evaluation import ModelSelector

class ActiveLearner:
    """Active learning implementation with support for multiple query strategies.
    
    Strategies:
        - uncertainty: Select instances with highest prediction uncertainty
        - entropy: Select instances with highest prediction entropy
        - margin: Select instances with smallest difference between top two predictions
        - density: Select instances from dense regions of feature space
        - qbc: Query by committee using multiple models
        - random: Random sampling from pool
    """
    
    SUPPORTED_STRATEGIES = ['uncertainty', 'entropy', 'margin', 'density', 'qbc', 'random']
    
    def __init__(self, 
                 model: Union[Estimator, List[Estimator]],
                 query_strategy: str = 'uncertainty',
                 n_neighbors: int = 5,
                 committee_size: int = 3,
                 random_state: Optional[int] = None):
        """Initialize active learner.
        
        Args:
            model: Base model(s) for making predictions
            query_strategy: Strategy for selecting instances
            n_neighbors: Number of neighbors for density estimation
            committee_size: Number of models for query by committee
            random_state: Random seed for reproducibility
        """
        if query_strategy == 'qbc':
            if not isinstance(model, list) or len(model) < 2:
                raise ValidationError("QBC requires multiple models")
            if not all(isinstance(m, Estimator) for m in model):
                raise ValidationError("All models must be instances of Estimator")
            self.models = model[:committee_size]
        else:
            if not isinstance(model, Estimator):
                raise ValidationError("Model must be an instance of Estimator")
            self.model = model
            
        if query_strategy not in self.SUPPORTED_STRATEGIES:
            raise ValidationError(
                f"Query strategy must be one of {self.SUPPORTED_STRATEGIES}"
            )
            
        self.query_strategy = query_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state or DEFAULT_RANDOM_STATE
        self.rng = np.random.RandomState(self.random_state)
        
    def _compute_density(self, X: np.ndarray) -> np.ndarray:
        """Compute density scores based on average distance to neighbors."""
        n_samples = len(X)
        densities = np.zeros(n_samples)
        
        for i in range(n_samples):
            distances = np.array([euclidean_distance(X[i], X[j]) for j in range(n_samples) if i != j])
            densities[i] = np.mean(np.sort(distances)[:self.n_neighbors])
            
        return 1 / (1 + densities)  # Convert distances to density scores
        
    def query(self, X_pool: np.ndarray, n_instances: int = 1) -> np.ndarray:
        """Select most informative instances for labeling.
        
        Args:
            X_pool: Array of unlabeled instances to select from
            n_instances: Number of instances to select
            
        Returns:
            Indices of selected instances
            
        Raises:
            ValidationError: If inputs are invalid
        """
        X_pool = check_array(X_pool)
        
        if n_instances < 1:
            raise ValidationError("n_instances must be greater than 0")
            
        if n_instances > len(X_pool):
            raise ValidationError(
                f"n_instances ({n_instances}) cannot be larger than "
                f"pool size ({len(X_pool)})"
            )
            
        try:
            if self.query_strategy == 'uncertainty':
                probas = self.model.predict_proba(X_pool)
                scores = 1 - np.max(probas, axis=1)
                
            elif self.query_strategy == 'entropy':
                probas = self.model.predict_proba(X_pool)
                scores = entropy(probas.T)
                
            elif self.query_strategy == 'margin':
                probas = self.model.predict_proba(X_pool)
                sorted_probas = np.sort(probas, axis=1)
                scores = sorted_probas[:,-1] - sorted_probas[:,-2]
                scores = 1 - scores  # Convert to uncertainty
                
            elif self.query_strategy == 'density':
                scores = self._compute_density(X_pool)
                
            elif self.query_strategy == 'qbc':
                all_preds = []
                for model in self.models:
                    probas = model.predict_proba(X_pool)
                    all_preds.append(probas)
                all_preds = np.array(all_preds)
                scores = np.mean(np.std(all_preds, axis=0), axis=1)
                
            else:  # Random sampling
                self.rng.seed(self.random_state)
                return self.rng.choice(len(X_pool), n_instances, replace=False)
                
            return np.argsort(scores)[-n_instances:]
            
        except Exception as e:
            # Fallback to random if strategy fails
            print(f"Warning: {self.query_strategy} failed, falling back to random sampling")
            self.rng.seed(self.random_state)
            return self.rng.choice(len(X_pool), n_instances, replace=False)