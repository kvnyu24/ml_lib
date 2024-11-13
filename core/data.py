"""Core data structures and dataset handling."""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import pickle
from pathlib import Path

@dataclass
class Dataset:
    """Container for dataset with features and labels."""
    X: np.ndarray  # Features
    y: np.ndarray  # Labels
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    
    def split(self, test_size: float = 0.2, val_size: float = 0.2,
             random_state: int = 42) -> Tuple['Dataset', 'Dataset', 'Dataset']:
        """Split into train, validation and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, 
            random_state=random_state, stratify=self.y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size,
            random_state=random_state, stratify=y_train
        )
        
        return (Dataset(X_train, y_train, self.feature_names, self.target_names),
                Dataset(X_val, y_val, self.feature_names, self.target_names),
                Dataset(X_test, y_test, self.feature_names, self.target_names))
                
    def select_features(self, k: int = 10) -> 'Dataset':
        """Select top k features using mutual information."""
        mi_scores = mutual_information(self.X, self.y)
        top_k_idx = np.argsort(mi_scores)[-k:]
        X_new = self.X[:, top_k_idx]
        selected_features = [self.feature_names[i] for i in top_k_idx] if self.feature_names else None
        return Dataset(X_new, self.y, selected_features, self.target_names)

def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2,
                    random_state: Optional[int] = None,
                    stratify: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
    """Split arrays into train and test subsets with optional stratification."""
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    
    if stratify is not None:
        # Get unique classes and their indices
        classes, class_indices = np.unique(stratify, return_inverse=True)
        
        # Split indices for each class proportionally
        train_idx = []
        test_idx = []
        
        for c in classes:
            c_idx = np.where(class_indices == c)[0]
            n_c_test = int(len(c_idx) * test_size)
            
            # Shuffle indices
            c_idx = np.random.permutation(c_idx)
            
            test_idx.extend(c_idx[:n_c_test])
            train_idx.extend(c_idx[n_c_test:])
            
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    else:
        indices = np.random.permutation(n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx])

def mutual_information(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate mutual information between features and target."""
    n_features = X.shape[1]
    mi_scores = np.zeros(n_features)
    
    # Calculate mutual information for each feature
    for i in range(n_features):
        mi_scores[i] = _mutual_info_feature(X[:, i], y)
        
    return mi_scores

def _mutual_info_feature(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Calculate mutual information between one feature and target."""
    # Discretize continuous feature into bins
    x_bins = np.histogram_bin_edges(x, bins=bins)
    x_discrete = np.digitize(x, x_bins[:-1])
    
    # Calculate probabilities
    x_vals, x_counts = np.unique(x_discrete, return_counts=True)
    y_vals, y_counts = np.unique(y, return_counts=True)
    
    p_x = x_counts / len(x)
    p_y = y_counts / len(y)
    
    # Calculate joint probability
    xy_counts = np.zeros((len(x_vals), len(y_vals)))
    for i, xi in enumerate(x_vals):
        for j, yj in enumerate(y_vals):
            xy_counts[i,j] = np.sum((x_discrete == xi) & (y == yj))
    
    p_xy = xy_counts / len(x)
    
    # Calculate mutual information
    mi = 0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            if p_xy[i,j] > 0:
                mi += p_xy[i,j] * np.log2(p_xy[i,j] / (p_x[i] * p_y[j]))
                
    return mi

def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']

def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """Save dataset to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)