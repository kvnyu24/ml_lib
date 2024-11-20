"""Decision tree implementations."""

import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass
from core import (
    Entropy,
    GiniImpurity,
    information_gain,
    Estimator,
    check_X_y
)

@dataclass
class Node:
    """Decision tree node."""
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[np.ndarray] = None
    is_leaf: bool = False

class DecisionTreeClassifier(Estimator):
    """Decision tree classifier implementation."""
    
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 criterion: str = 'gini',
                 max_features: Optional[Union[int, float, str]] = None,
                 random_state: Optional[int] = None):
        """Initialize decision tree classifier.
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf
            criterion: Split criterion ('gini' or 'entropy')
            max_features: Number of features to consider for best split
            random_state: Random state for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        
        self.tree_: Optional[Node] = None
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        
        self.criterion_func = (GiniImpurity() if criterion == 'gini' 
                             else Entropy())
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """Build decision tree classifier.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            self: Fitted estimator
        """
        X, y = check_X_y(X, y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        # Determine max_features
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features_ = int(np.sqrt(self.n_features_))
            elif self.max_features == 'log2':
                self.max_features_ = int(np.log2(self.n_features_))
        elif isinstance(self.max_features, float):
            self.max_features_ = int(self.max_features * self.n_features_)
        elif isinstance(self.max_features, int):
            self.max_features_ = self.max_features
        else:
            self.max_features_ = self.n_features_
            
        self.tree_ = self._grow_tree(X, y)
        return self
        
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, 
                   depth: int = 0) -> Node:
        """Recursively grow decision tree.
        
        Args:
            X: Training data
            y: Target values
            depth: Current depth
            
        Returns:
            Node: Root node of tree/subtree
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            assert self.n_classes_ is not None, "n_classes_ not set. Call fit first."
            return Node(
                value=np.bincount(y, minlength=self.n_classes_).astype(float) / n_samples,
                is_leaf=True
            )
            
        # Find best split
        feature_indices = np.random.choice(
            n_features, 
            size=self.max_features_,
            replace=False
        )
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                    
                gain = information_gain(
                    y,
                    [y[left_mask], y[right_mask]],
                    self.criterion_func
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        if best_gain == -1:  # No valid split found
            assert self.n_classes_ is not None, "n_classes_ not set. Call fit first."
            return Node(
                value=np.bincount(y, minlength=self.n_classes_).astype(float) / n_samples,
                is_leaf=True
            )
            
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Class probabilities
        """
        if self.tree_ is None:
            raise ValueError("Call fit before predict")
        return np.array([self._traverse_tree(x, self.tree_) for x in X])
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        return np.argmax(self.predict_proba(X), axis=1)
        
    def _traverse_tree(self, x: np.ndarray, node: Node) -> np.ndarray:
        """Traverse tree to make prediction.
        
        Args:
            x: Single input sample
            node: Current tree node
            
        Returns:
            Class probabilities
        """
        if node.is_leaf:
            assert node.value is not None
            return node.value
            
        assert node.feature_idx is not None
        assert node.threshold is not None
        assert node.left is not None
        assert node.right is not None
            
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class DecisionTreeRegressor(Estimator):
    """Decision tree regressor implementation."""
    
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[int, float, str]] = None,
                 random_state: Optional[int] = None):
        """Initialize decision tree regressor.
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf
            max_features: Number of features to consider for best split
            random_state: Random state for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        self.tree_: Optional[Node] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """Build decision tree regressor.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            self: Fitted estimator
        """
        X, y = check_X_y(X, y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.n_features_ = X.shape[1]
        
        # Determine max_features
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features_ = int(np.sqrt(self.n_features_))
            elif self.max_features == 'log2':
                self.max_features_ = int(np.log2(self.n_features_))
        elif isinstance(self.max_features, float):
            self.max_features_ = int(self.max_features * self.n_features_)
        elif isinstance(self.max_features, int):
            self.max_features_ = self.max_features
        else:
            self.max_features_ = self.n_features_
            
        self.tree_ = self._grow_tree(X, y)
        return self
        
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, 
                   depth: int = 0) -> Node:
        """Recursively grow decision tree.
        
        Args:
            X: Training data
            y: Target values
            depth: Current depth
            
        Returns:
            Node: Root node of tree/subtree
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split):
            return Node(
                value=np.array([np.mean(y)]),
                is_leaf=True
            )
            
        # Find best split
        feature_indices = np.random.choice(
            n_features, 
            size=self.max_features_,
            replace=False
        )
        
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                    
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                mse = (np.sum((y_left - np.mean(y_left))**2) +
                      np.sum((y_right - np.mean(y_right))**2))
                
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        if best_feature is None:  # No valid split found
            return Node(
                value=np.array([np.mean(y)]),
                is_leaf=True
            )
            
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values
        """
        if self.tree_ is None:
            raise ValueError("Call fit before predict")
        return np.array([self._traverse_tree(x, self.tree_)[0] for x in X])
        
    def _traverse_tree(self, x: np.ndarray, node: Node) -> np.ndarray:
        """Traverse tree to make prediction.
        
        Args:
            x: Single input sample
            node: Current tree node
            
        Returns:
            Predicted value
        """
        if node.is_leaf:
            assert node.value is not None
            return node.value
            
        assert node.feature_idx is not None
        assert node.threshold is not None
        assert node.left is not None
        assert node.right is not None
            
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)