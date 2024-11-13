"""Core data structures and dataset handling."""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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