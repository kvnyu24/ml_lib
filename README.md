# ML Library

A comprehensive machine learning library implementing various algorithms, preprocessing techniques, and utilities for data science and machine learning tasks. This library is implemented from scratch with help of LLMs to help everyone better understand the inner workings of machine learning models. Feel free to fork and contribute! Will gradually add more models, features, and utilities, as well as documentation and examples.

## 🌟 Features

### Core Components
- **Base Classes**: Extensible base classes for estimators, optimizers, losses, layers, and transformers
- **Configuration Management**: Flexible configuration system for models and training
- **Validation**: Robust input validation and error checking
- **Metrics**: Common evaluation metrics and custom metric support
- **Callbacks**: Training callbacks including early stopping and model checkpoints
- **Logging**: Comprehensive logging system for training and evaluation

### Preprocessing
- **Scalers**
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
  - QuantileScaler

- **Encoders**
  - LabelEncoder
  - OneHotEncoder
  - OrdinalEncoder
  - TargetEncoder

- **Feature Engineering**
  - PolynomialFeatures
  - InteractionFeatures
  - CustomFeatureTransformer

- **Missing Value Handling**
  - MissingValueImputer
  - KNNImputer
  - TimeSeriesImputer

- **Specialized Preprocessing**
  - Text Processing (TF-IDF, Word2Vec)
  - Image Processing
  - Audio Processing
  - Graph Data Processing
  - Time Series Processing
  - Sequence Data Processing

### Optimization
- **Gradient-Based Optimizers**
  - SGD (with momentum)
  - Adam
  - RMSprop
  - Adagrad
  - Adamax
  - Nadam

- **Global Optimizers**
  - Particle Swarm Optimization (PSO)
  - CMA-ES
  - Nelder-Mead

### Models
- **Linear Models**
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Elastic Net

- **Neural Networks**
  - Multi-Layer Perceptron
  - Custom Layer Support
  - Various Activation Functions

- **Support Vector Machines**
  - Linear SVM
  - Kernel SVM

### Model Selection & Evaluation
- Cross-validation utilities
- Hyperparameter optimization
- Model selection tools
- Performance metrics

## 📦 Installation

```bash
pip install mllib
```

## 🚀 Quick Start

```python
from mllib.preprocessing import StandardScaler, OneHotEncoder
from mllib.models import LinearRegression
from mllib.pipeline import Pipeline

# Create a preprocessing and modeling pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('encoder', OneHotEncoder()),
    ('model', LinearRegression())
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## 📖 Documentation

### Preprocessing Example

```python
from mllib.preprocessing import (
    StandardScaler,
    TextPreprocessor,
    MissingValueImputer
)

# Handle missing values
imputer = MissingValueImputer(strategy='mean')
X_clean = imputer.fit_transform(X)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Process text data
text_processor = TextPreprocessor(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True
)
text_features = text_processor.fit_transform(text_data)
```

### Model Training Example

```python
from mllib.models import MLP
from mllib.optimization import Adam
from mllib.callbacks import EarlyStopping

# Create and configure model
model = MLP(
    hidden_layers=[64, 32],
    activation='relu',
    optimizer=Adam(learning_rate=0.001)
)

# Add callbacks
callbacks = [
    EarlyStopping(patience=5, min_delta=1e-4)
]

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

## 🛠️ Project Structure

```
mllib/
├── core/
│   ├── base.py          # Base classes
│   ├── config.py        # Configuration management
│   ├── exceptions.py    # Custom exceptions
│   ├── validation.py    # Input validation
│   └── metrics.py       # Evaluation metrics
├── preprocessing/
│   ├── scalers.py       # Data scaling
│   ├── encoders.py      # Feature encoding
│   ├── text.py          # Text processing
│   ├── image.py         # Image processing
│   └── audio.py         # Audio processing
├── optimization/
│   ├── gradient.py      # Gradient-based optimizers
│   └── global_opt.py    # Global optimization
├── models/
│   ├── linear.py        # Linear models
│   ├── neural.py        # Neural networks
│   └── svm.py           # Support vector machines
└── utils/
    ├── visualization.py # Plotting utilities
    └── io.py           # Input/output utilities
```