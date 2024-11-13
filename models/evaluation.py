"""Model evaluation and selection utilities."""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from core import Estimator, Metric
from core.data import Dataset
from core.validation import check_X_y, check_is_fitted
from core.metrics import MetricList, get_metric
from core.logging import get_logger

logger = get_logger(__name__)

class CrossValidator:
    """Base class for cross-validation splitters."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Generate indices to split data into training and validation."""
        raise NotImplementedError

class KFoldCV(CrossValidator):
    """K-Fold cross-validation splitter."""
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
            
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, val_indices
            current = stop

class StratifiedKFoldCV(CrossValidator):
    """Stratified K-Fold cross-validation splitter."""
    
    def split(self, X: np.ndarray, y: np.ndarray):
        unique_classes = np.unique(y)
        class_indices = [np.where(y == c)[0] for c in unique_classes]
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            for idx in class_indices:
                rng.shuffle(idx)
                
        splits = []
        for class_idx in class_indices:
            splits.append(np.array_split(class_idx, self.n_splits))
            
        for fold in range(self.n_splits):
            val_indices = np.concatenate([split[fold] for split in splits])
            train_indices = np.concatenate([
                np.concatenate([split[i] for i in range(self.n_splits) if i != fold])
                for split in splits
            ])
            yield train_indices, val_indices

class ModelEvaluator:
    """Model evaluation utilities."""
    
    @staticmethod
    def cross_validate(model: Estimator,
                      X: np.ndarray,
                      y: np.ndarray,
                      metrics: Dict[str, Union[str, Metric, Callable]],
                      cv: int = 5,
                      stratify: bool = True) -> Dict[str, List[float]]:
        """Perform cross-validation with multiple metrics."""
        cv_splitter = StratifiedKFoldCV(n_splits=cv) if stratify else KFoldCV(n_splits=cv)
        results = {name: [] for name in metrics}
        
        for train_idx, val_idx in cv_splitter.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            for name, metric in metrics.items():
                if isinstance(metric, str):
                    metric = get_metric(metric)
                score = metric(y_val, y_pred)
                results[name].append(score)
                
        return results

    @staticmethod
    def cross_validate_dataset(model: Any,
                             dataset: Dataset,
                             metrics: Optional[List[Union[str, Callable]]] = None,
                             validation_strategy: str = 'kfold',
                             n_splits: int = 5,
                             test_size: float = 0.2,
                             stratify: bool = True,
                             shuffle: bool = True,
                             random_state: Optional[int] = None) -> Dict[str, List[float]]:
        """Perform cross-validation on a dataset."""
        X, y = check_X_y(dataset.X, dataset.y)
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_list = MetricList(metrics)
        
        cv_options = {
            'kfold': KFoldCV(n_splits=n_splits, shuffle=shuffle, random_state=random_state),
            'stratified': StratifiedKFoldCV(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        }
        
        if validation_strategy not in cv_options:
            raise ValueError(f"Unknown validation strategy: {validation_strategy}")
            
        cv = cv_options[validation_strategy]
        results = {metric.name: [] for metric in metric_list.metrics}
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                model.fit(X_train, y_train)
                check_is_fitted(model)
                
                y_pred = model.predict(X_val)
                fold_metrics = metric_list(y_val, y_pred)
                
                for metric_name, value in fold_metrics.items():
                    results[metric_name].append(value)
                    
                logger.info(f"Fold {fold + 1}/{n_splits} completed")
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                raise
                
        return results

class ModelSelector:
    """Model selection and hyperparameter optimization."""
    
    def __init__(self,
                 model: Estimator,
                 param_grid: Dict,
                 scoring: Union[str, Callable] = 'accuracy',
                 cv: int = 5):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_model_ = None
        self.best_params_ = None
        self.best_score_ = None
        
    def _get_param_combinations(self):
        """Get all combinations of parameters."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelSelector':
        """Find best hyperparameters using grid search."""
        if isinstance(self.scoring, str):
            self.scoring = get_metric(self.scoring)
            
        best_score = float('-inf')
        cv = StratifiedKFoldCV(n_splits=self.cv)
        
        for params in self._get_param_combinations():
            self.model.set_params(**params)
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_val)
                scores.append(self.scoring(y_val, y_pred))
                
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = best_score
                
        self.model.set_params(**self.best_params_)
        self.model.fit(X, y)
        self.best_model_ = self.model
        
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self