"""Audio preprocessing utilities."""

import numpy as np
from typing import Optional, Dict, Union
from core import BaseTransformer
import librosa

class AudioPreprocessor(BaseTransformer):
    """Audio signal preprocessing utilities."""
    
    def __init__(self,
                 sample_rate: int = 22050,
                 duration: Optional[float] = None,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AudioPreprocessor':
        """Fit preprocessor (no-op)."""
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform audio signals to mel spectrograms."""
        spectrograms = []
        
        for signal in X:
            if self.duration:
                signal = signal[:int(self.sample_rate * self.duration)]
                
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=signal,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            spectrograms.append(mel_spec)
            
        return np.array(spectrograms)

class AudioFeatureExtractor(BaseTransformer):
    """Extract audio features."""
    
    def __init__(self,
                 features: List[str] = ['mfcc', 'spectral_centroid', 'zero_crossing_rate'],
                 n_mfcc: int = 13,
                 sample_rate: int = 22050):
        self.features = features
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AudioFeatureExtractor':
        """Fit extractor (no-op)."""
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract audio features."""
        features_list = []
        
        for signal in X:
            features_dict = {}
            
            if 'mfcc' in self.features:
                mfcc = librosa.feature.mfcc(
                    y=signal,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc
                )
                features_dict['mfcc'] = np.mean(mfcc, axis=1)
                
            if 'spectral_centroid' in self.features:
                centroid = librosa.feature.spectral_centroid(
                    y=signal,
                    sr=self.sample_rate
                )
                features_dict['spectral_centroid'] = np.mean(centroid)
                
            if 'zero_crossing_rate' in self.features:
                zcr = librosa.feature.zero_crossing_rate(signal)
                features_dict['zero_crossing_rate'] = np.mean(zcr)
                
            # Concatenate all features
            features_list.append(np.concatenate([
                v if isinstance(v, np.ndarray) else np.array([v])
                for v in features_dict.values()
            ]))
            
        return np.array(features_list) 