"""Example of using ElasticNet regression with cross-validation."""

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.linear import ElasticNetRegression, ElasticNetLoss
from preprocessing import StandardScaler
from core import Dataset
from utils.visualization import plot_learning_curves
from models.evaluation import ModelEvaluator

def generate_regression_data():
    """Generate synthetic regression dataset."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    # Validate generated data
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Failed to generate valid regression data")
    return Dataset(X, y)

def main():
    # Generate dataset
    dataset = generate_regression_data()
    
    # Add validation check after dataset generation
    if dataset.X.shape[0] == 0 or dataset.y.shape[0] == 0:
        raise ValueError("Generated dataset is empty")
    
    # Split into train/val/test sets
    train_data, val_data, test_data = dataset.split(
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    print("Train Data:", train_data, "Val Data:", val_data, "Test Data:", test_data)
    
    # Add validation checks after splitting
    if train_data.X.shape[0] == 0:
        raise ValueError("Training set is empty after splitting")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data.X)
    X_val = scaler.transform(val_data.X)
    X_test = scaler.transform(test_data.X)
    
    # Initialize model
    model = ElasticNetRegression(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=1000,
        tol=1e-4
    )
    
    # Train and evaluate
    history = model.fit(
        X_train, train_data.y,
        validation_data=(X_val, val_data.y)
    )
    
    # Add validation check for history
    if not isinstance(history, dict) or not history:
        print("Warning: No training history available")
    else:
        # Plot learning curves
        plot_learning_curves(
            history,
            title='ElasticNet Regression Training History',
            save_path='elastic_net_history.png'
        )
    
    # Evaluate on test set
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(
        model,
        X_test,
        test_data.y,
        metrics=['mse', 'mae', 'r2']
    )
    
    print("\nTest Set Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main() 