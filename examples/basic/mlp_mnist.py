"""Basic MLP example using MNIST dataset."""

import numpy as np
import sys
import os
from sklearn.datasets import fetch_openml

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.neural import MLP, Dense, CrossEntropyLoss, ReLU, Softmax
from core import EarlyStopping
from preprocessing import StandardScaler, OneHotEncoder

def load_mnist():
    """Load MNIST dataset from OpenML."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype('float32')
    
    # Normalize pixel values to [0,1]
    X /= 255.0
    
    # Convert labels to one-hot encoding
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1))
    
    return X, y

def main():
    # Load and preprocess data
    X, y = load_mnist()
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train/validation sets
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Create layers
    layers = [
        Dense(784, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10),
        Softmax()
    ]
    
    # Create model with early stopping
    model = MLP(
        layers=layers,
        loss_function=CrossEntropyLoss,
        optimizer={'type': 'adam', 'learning_rate': 0.001},
        metrics=[EarlyStopping(patience=5, min_delta=0.001)]
    )

    # Train model
    model.fit(X_train, y_train, 
             batch_size=128,
             epochs=20,
             validation_data=(X_val, y_val))
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_loss = CrossEntropyLoss.forward(val_pred, y_val)
    val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    main()