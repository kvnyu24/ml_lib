"""Training callbacks for monitoring and control."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import time

class Callback(ABC):
    """Base class for callbacks."""
    
    def __init__(self):
        self.__name__ = self.__class__.__name__
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass

class EarlyStopping(Callback):
    """Stop training when metric stops improving."""
    
    def __init__(self, monitor: str = 'val_loss', 
                 patience: int = 0,
                 min_delta: float = 0.0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best = np.Inf
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

class ModelCheckpoint(Callback):
    """Save model after every epoch."""
    
    def __init__(self, filepath: str, 
                 monitor: str = 'val_loss',
                 save_best_only: bool = False):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = np.Inf
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.save_best_only:
            if current < self.best:
                self.best = current
                self.model.save(self.filepath)
        else:
            self.model.save(f"{self.filepath}_epoch_{epoch}")