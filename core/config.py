"""Configuration management."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import json
from pathlib import Path
from .exceptions import ConfigurationError
from .dtypes import (
    DEFAULT_OPTIMIZER_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_LOSSES
)

@dataclass 
class ModelConfig:
    """Model configuration."""
    input_dim: int
    output_dim: int
    hidden_dims: Optional[List[int]] = None
    activation: str = 'relu'
    loss: str = 'mse'
    optimizer: str = 'adam'
    random_state: Optional[int] = None
    device: str = 'cpu'
    dtype: str = 'float32'
    verbose: bool = False
    
    def __post_init__(self):
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ConfigurationError(f"Unsupported activation: {self.activation}")
        if self.loss not in SUPPORTED_LOSSES:
            raise ConfigurationError(f"Unsupported loss: {self.loss}")
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ConfigurationError(f"Unsupported optimizer: {self.optimizer}")

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32  # Default value instead of referencing dict
    epochs: int = 100
    validation_split: float = 0.2
    shuffle: bool = True
    verbose: bool = True
    early_stopping: bool = True
    patience: int = 5
    
    def __post_init__(self):
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be positive")
        if self.epochs < 1:
            raise ConfigurationError("epochs must be positive") 
        if not 0 <= self.validation_split < 1:
            raise ConfigurationError("validation_split must be in [0, 1)")

class ConfigManager:
    """Configuration management utilities."""
    
    @staticmethod
    def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
        filepath = Path(filepath)
        if not filepath.exists():
            raise ConfigurationError(f"Config file not found: {filepath}")
            
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise ConfigurationError(f"Error loading config: {e}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            raise ConfigurationError(f"Error saving config: {e}")