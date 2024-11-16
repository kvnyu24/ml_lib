"""Example of using SVM classifier with kernel selection."""

import numpy as np
from sklearn.datasets import make_classification
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.svm import SVM, SVMModelSelector, AdvancedKernels
from preprocessing import StandardScaler
from core import Dataset
from utils.visualization import plot_decision_boundary
from models.evaluation import ModelEvaluator

def generate_nonlinear_data():
    """Generate synthetic nonlinear classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=2,  # 2D for visualization
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=2,
        class_sep=1.0
    )
    return Dataset(X, y)

def main():
    # Generate dataset
    dataset = generate_nonlinear_data()
    
    # Split data
    train_data, val_data, test_data = dataset.split(
        test_size=0.2,
        val_size=0.2
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data.X)
    X_val = scaler.transform(val_data.X)
    X_test = scaler.transform(test_data.X)
    
    # Define kernels to try
    kernels = {
        'rbf': {'gamma': [0.1, 1.0, 10.0]},
        'polynomial': {'degree': [2, 3, 4]},
        'sigmoid': {'gamma': [0.1, 1.0], 'coef0': [0, 1]}
    }
    
    # Initialize model selector
    model_selector = SVMModelSelector(
        base_model=SVM(kernel='rbf', C=1.0),
        param_distributions=kernels,
        cv=5,
        scoring='accuracy'
    )
    
    # Find best model
    best_model = model_selector.select_best_model(X_train, train_data.y)
    
    # Evaluate on test set
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(
        best_model,
        X_test,
        test_data.y,
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    print("\nBest Model Parameters:", model_selector.best_params_)
    print("\nTest Set Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Plot decision boundary
    plot_decision_boundary(
        best_model,
        X_test,
        test_data.y,
        title='SVM Decision Boundary',
        save_path='svm_boundary.png'
    )

if __name__ == '__main__':
    main() 