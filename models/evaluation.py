"""Model evaluation and selection utilities."""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    LeavePOut, TimeSeriesSplit, ShuffleSplit, cross_validate,
    GridSearchCV
)
from sklearn.metrics import make_scorer
from core import Estimator, Metric
from core.data import Dataset
from core.validation import check_X_y, check_is_fitted
from core.metrics import MetricList, get_metric
from core.logging import get_logger

logger = get_logger(__name__)

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
        cv = StratifiedKFold(n_splits=cv) if stratify else KFold(n_splits=cv)
            
        # Convert metrics to scorers
        scorers = {
            name: make_scorer(metric) if callable(metric) else metric
            for name, metric in metrics.items()
        }
        
        # Perform cross-validation
        scores = cross_validate(
            model, X, y,
            scoring=scorers,
            cv=cv,
            n_jobs=-1
        )
        
        return {
            name: scores[f'test_{name}']
            for name in metrics.keys()
        }

    @staticmethod
    def cross_validate_dataset(model: Any,
                             dataset: Dataset,
                             metrics: Optional[List[Union[str, Callable]]] = None,
                             validation_strategy: str = 'kfold',
                             n_splits: int = 5,
                             test_size: float = 0.2,
                             stratify: bool = True,
                             shuffle: bool = True,
                             random_state: Optional[int] = None,
                             leave_p_out: int = 1) -> Dict[str, List[float]]:
        """Perform cross-validation on a dataset.
        
        Args:
            model: Model instance with fit and predict methods
            dataset: Dataset instance containing features and targets
            metrics: List of metrics to evaluate. Can be strings or callables.
                    If None, uses default metrics.
            validation_strategy: One of ['kfold', 'stratified', 'loo', 'lpo', 
                                       'timeseries', 'shuffle']
            n_splits: Number of cross-validation folds (for KFold strategies)
            test_size: Test size fraction (for ShuffleSplit)
            stratify: Whether to preserve class distribution in folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            leave_p_out: Number of samples to leave out (for LeavePOut)
            
        Returns:
            Dictionary mapping metric names to lists of scores for each fold
        """
        X, y = check_X_y(dataset.X, dataset.y)
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_list = MetricList(metrics)
        
        # Initialize cross-validation splitter
        cv_options = {
            'kfold': StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) 
                    if stratify else 
                    KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state),
            'loo': LeaveOneOut(),
            'lpo': LeavePOut(p=leave_p_out),
            'timeseries': TimeSeriesSplit(n_splits=n_splits),
            'shuffle': ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        }
        
        if validation_strategy not in cv_options:
            raise ValueError(f"Unknown validation strategy: {validation_strategy}")
            
        cv = cv_options[validation_strategy]
        results = {metric.name: [] for metric in metric_list.metrics}
        
        # Perform cross-validation
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
                    
                logger.info(f"Fold {fold + 1}/{n_splits if hasattr(cv, 'n_splits') else 'N'} completed")
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                raise
                
        return results

class ModelSelector:
    """Model selection and hyperparameter optimization."""
    
    def __init__(self,
                 model: Estimator,
                 param_grid: Dict,
                 scoring: Union[str, Callable, List[str]] = 'accuracy',
                 cv: int = 5,
                 n_jobs: int = -1):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_model_ = None
        self.best_params_ = None
        self.best_score_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelSelector':
        """Find best hyperparameters using grid search."""
        grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs
        )
        
        grid_search.fit(X, y)
        
        self.best_model_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best score: {self.best_score_:.4f}")
        
        return self

class CrossValidationSplitter:
    """Cross-validation split utilities."""
    
    @staticmethod
    def get_cv_splits(X: np.ndarray,
                     y: np.ndarray,
                     n_splits: int = 5,
                     stratify: bool = True,
                     shuffle: bool = True,
                     random_state: Optional[int] = None) -> List[tuple]:
        """Get cross-validation splits."""
        cv = StratifiedKFold if stratify else KFold
        splitter = cv(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(splitter.split(X, y))