"""Core data structures and dataset handling."""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(self.X, self.y)
        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        return Dataset(X_new, self.y, selected_features, self.target_names) 

def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2,
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """Split arrays into train and test subsets."""
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx])

def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']

def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """Save dataset to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)