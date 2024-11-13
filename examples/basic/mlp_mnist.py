"""Basic MLP example using MNIST dataset."""

import numpy as np
import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.neural import MLP
from core import EarlyStopping
from preprocessing import StandardScaler

def load_mnist():
    """Dummy function to load MNIST - replace with actual data loading."""
    # Simulated MNIST data
    X = np.random.randn(1000, 784)  # 1000 images, 28x28 flattened
    y = np.random.randint(0, 10, 1000)  # 10 classes
    return X, y

def main():
    # Load and preprocess data
    X, y = load_mnist()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create and train model
    model = MLP(
        hidden_layers=[128, 64],
        activation='relu',
        output_activation='softmax',
        loss='categorical_crossentropy'
    )
    
    callbacks = [EarlyStopping(patience=5)]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )


    print(history)

if __name__ == '__main__':
    main()