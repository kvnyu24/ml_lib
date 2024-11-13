"""Input/output utilities."""

import pickle
import json
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_model(model: Any, path: str, compress: bool = True) -> None:
    """Save model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(path: str) -> Any:
    """Load model from disk."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def save_data(data: Any, path: str, format: str = 'pickle') -> None:
    """Save data to disk in specified format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'json':
            with open(path, 'w') as f:
                json.dump(data, f)
        elif format == 'numpy':
            np.save(path, data)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Data saved to {path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def load_data(path: str, format: str = 'pickle') -> Any:
    """Load data from disk."""
    try:
        if format == 'pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif format == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif format == 'numpy':
            data = np.load(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Data loaded from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise 