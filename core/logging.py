"""Logging configuration and utilities."""

import logging
import sys
from typing import Optional
from pathlib import Path

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get or create logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.propagate = False
        
    return logger

class TrainingLogger:
    """Logger for model training progress."""
    
    def __init__(self, model_name: str, log_dir: Optional[str] = None):
        self.model_name = model_name
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(f"training.{model_name}")
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'val_loss': [],
            'metrics': []
        }
        
    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for an epoch."""
        msg = f"Epoch {epoch:3d}"
        for metric, value in metrics.items():
            msg += f" - {metric}: {value:.4f}"
            self.history.setdefault(metric, []).append(value)
        self.logger.info(msg) 