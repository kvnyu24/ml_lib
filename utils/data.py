"""Data loading and processing utilities."""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import pickle

def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2,
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx])

def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']

def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f) 