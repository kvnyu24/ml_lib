"""Random forest implementations."""

import numpy as np
from typing import Optional, List, Union
from joblib import Parallel, delayed
from .decision_tree import DecisionTreeClassifier
from core import (
    Estimator,
    check_X_y,
    check_array,
    check_is_fitted,
    get_logger,
    ValidationError
)

class RandomForestClassifier(Estimator):
    """Random forest classifier implementation."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, float, str]] = 'sqrt',
                 bootstrap: bool = True,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None):
        """Initialize random forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap sampling
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.estimators_ = []
        self.n_classes_ = None
        self.n_features_ = None
        
    def _make_estimator(self, random_state: Optional[int] = None) -> DecisionTreeClassifier:
        """Make a single decision tree estimator."""
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=random_state
        )
        
    def _parallel_build_trees(self, X: np.ndarray, y: np.ndarray, 
                            random_state: Optional[int] = None) -> List[DecisionTreeClassifier]:
        """Build trees in parallel."""
        n_samples = X.shape[0]
        
        if random_state is not None:
            np.random.seed(random_state)
            
        trees = []
        for i in range(self.n_estimators):
            tree = self._make_estimator(random_state=random_state + i if random_state else None)
            
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                sample_X = X[indices]
                sample_y = y[indices]
            else:
                sample_X = X
                sample_y = y
                
            tree.fit(sample_X, sample_y)
            trees.append(tree)
            
        return trees
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """Build random forest classifier.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            self: Fitted estimator
        """
        X, y = check_X_y(X, y)
        
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        # Build trees in parallel
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._parallel_build_trees)(
                X, y, self.random_state + i if self.random_state else None
            )
            for i in range(self.n_estimators)
        )
        
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Class probabilities
        """
        # Parallel prediction
        all_proba = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict_proba)(X)
            for tree in self.estimators_
        )
        
        # Average predictions
        return np.mean(all_proba, axis=0)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        return np.argmax(self.predict_proba(X), axis=1) 