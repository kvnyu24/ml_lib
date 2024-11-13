"""Sequence data preprocessing utilities."""

import numpy as np
from typing import Optional, List, Dict, Union
from core import BaseTransformer

class SequencePadder(BaseTransformer):
    """Pad sequences to equal length."""
    
    def __init__(self, max_length: Optional[int] = None, padding: str = 'post', value: float = 0.0):
        self.max_length = max_length
        self.padding = padding
        self.value = value
        self._max_seq_length = None
        
    def fit(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> 'SequencePadder':
        """Compute max sequence length if not specified."""
        if self.max_length is None:
            self._max_seq_length = max(len(seq) for seq in X)
        else:
            self._max_seq_length = self.max_length
        return self
        
    def transform(self, X: List[np.ndarray]) -> np.ndarray:
        """Pad sequences."""
        padded = []
        
        for seq in X:
            if len(seq) > self._max_seq_length:
                padded_seq = seq[:self._max_seq_length]
            else:
                pad_width = self._max_seq_length - len(seq)
                if self.padding == 'post':
                    padded_seq = np.pad(seq, ((0, pad_width), (0, 0)), 
                                      constant_values=self.value)
                else:  # pre-padding
                    padded_seq = np.pad(seq, ((pad_width, 0), (0, 0)), 
                                      constant_values=self.value)
            padded.append(padded_seq)
            
        return np.array(padded)

class SequenceAugmenter(BaseTransformer):
    """Augment sequence data."""
    
    def __init__(self,
                 noise_level: float = 0.1,
                 time_warp_factor: float = 0.2,
                 crop_ratio: float = 0.1):
        self.noise_level = noise_level
        self.time_warp_factor = time_warp_factor
        self.crop_ratio = crop_ratio
        
    def fit(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> 'SequenceAugmenter':
        """Fit augmenter (no-op)."""
        return self
        
    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """Apply sequence augmentations."""
        augmented = []
        
        for seq in X:
            # Add random noise
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, seq.shape)
                seq = seq + noise
                
            # Time warping
            if self.time_warp_factor > 0:
                time_steps = len(seq)
                warp = np.random.uniform(1-self.time_warp_factor, 
                                       1+self.time_warp_factor, 
                                       time_steps)
                seq = seq * warp[:, np.newaxis]
                
            # Random cropping
            if self.crop_ratio > 0:
                crop_size = int(len(seq) * (1 - self.crop_ratio))
                start = np.random.randint(0, len(seq) - crop_size)
                seq = seq[start:start+crop_size]
                
            augmented.append(seq)
            
        return augmented 