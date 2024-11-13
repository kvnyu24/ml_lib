"""
Advanced Fourier Series and Spectral Learning Library
==================================================

A comprehensive library for Fourier-based machine learning and spectral analysis:

- Adaptive Fourier series models with regularization
- Spectral kernel methods and feature learning
- Frequency domain optimization
- Fourier neural networks
- Time-frequency analysis tools
- Spectral clustering extensions
- Parametric and differential equation solvers
- Neural ODEs and dynamical systems

Complements existing ML algorithms with specialized Fourier-based methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import fftpack, integrate
from scipy.optimize import minimize
from core import (
    Estimator,
    Transformer,
    check_array,
    check_X_y,
    check_is_fitted,
    Number,
    Array,
    Features,
    Target
)

@dataclass
class Dataset:
    """Container for dataset."""
    x: np.ndarray
    y: np.ndarray
    freq_domain: Optional[np.ndarray] = None

class BaseFourierModel(Estimator):
    """Abstract base class for Fourier series models."""
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data."""
        pass
        
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

class AdaptiveFourierModel(BaseFourierModel):
    """Fourier model with adaptive frequency selection."""
    
    def __init__(self, max_components: int = 20, 
                 alpha: float = 0.1,
                 freq_init: str = 'uniform'):
        self.max_components = max_components
        self.alpha = alpha # Regularization strength
        self.freq_init = freq_init
        self.frequencies = None
        self.amplitudes = None
        self.phases = None
        
    def _initialize_frequencies(self, n_samples: int):
        if self.freq_init == 'uniform':
            self.frequencies = np.linspace(0, n_samples//2, self.max_components)
        elif self.freq_init == 'random':
            self.frequencies = np.random.uniform(0, n_samples//2, self.max_components)
            
    def _objective(self, params, x, y):
        amplitudes = params[:self.max_components]
        phases = params[self.max_components:]
        
        y_pred = np.zeros_like(x)
        for i, (a, p, f) in enumerate(zip(amplitudes, phases, self.frequencies)):
            y_pred += a * np.sin(2*np.pi*f*x + p)
            
        # L1 regularization on amplitudes
        reg_term = self.alpha * np.sum(np.abs(amplitudes))
        return np.mean((y - y_pred)**2) + reg_term
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x, y = check_X_y(x, y)
        self._initialize_frequencies(len(x))
        
        # Initialize amplitudes and phases
        params_init = np.random.randn(2*self.max_components)
        
        # Optimize amplitudes and phases
        result = minimize(self._objective, params_init, args=(x, y),
                        method='L-BFGS-B')
        
        self.amplitudes = result.x[:self.max_components]
        self.phases = result.x[self.max_components:]
        return self
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['amplitudes', 'phases'])
        x = check_array(x)
            
        y_pred = np.zeros_like(x)
        for a, p, f in zip(self.amplitudes, self.phases, self.frequencies):
            y_pred += a * np.sin(2*np.pi*f*x + p)
        return y_pred

class SpectralKernel:
    """Kernel functions in frequency domain."""
    
    @staticmethod
    def frequency_rbf(x1: np.ndarray, x2: np.ndarray, gamma: float) -> np.ndarray:
        """RBF kernel in frequency domain."""
        x1_fft = fftpack.fft(x1)
        x2_fft = fftpack.fft(x2)
        return np.exp(-gamma * np.abs(x1_fft - x2_fft)**2)
        
    @staticmethod
    def frequency_linear(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Linear kernel in frequency domain."""
        x1_fft = fftpack.fft(x1)
        x2_fft = fftpack.fft(x2)
        return np.real(x1_fft @ x2_fft.conj())

class FourierFeatureLearner(Transformer):
    """Learn optimal Fourier features."""
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.feature_frequencies = None
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Learn and transform to Fourier features."""
        X = check_array(X)
        
        # Compute frequency components
        freqs = fftpack.fftfreq(len(X))
        fft_vals = fftpack.fft(X)
        
        # Select top frequencies by magnitude
        magnitude_order = np.argsort(np.abs(fft_vals))[::-1]
        self.feature_frequencies = freqs[magnitude_order[:self.n_components]]
        
        # Transform to Fourier features
        features = np.zeros((len(X), 2*self.n_components))
        for i, freq in enumerate(self.feature_frequencies):
            features[:,2*i] = np.cos(2*np.pi*freq*X)
            features[:,2*i+1] = np.sin(2*np.pi*freq*X)
        return features

class TimeFrequencyAnalyzer:
    """Time-frequency analysis tools."""
    
    @staticmethod
    def compute_spectrogram(x: np.ndarray, 
                          window_size: int,
                          hop_length: int) -> np.ndarray:
        """Compute spectrogram using sliding window FFT."""
        n_windows = (len(x) - window_size) // hop_length + 1
        spectrogram = np.zeros((window_size//2 + 1, n_windows))
        
        for i in range(n_windows):
            start = i * hop_length
            window = x[start:start + window_size]
            spectrogram[:,i] = np.abs(fftpack.rfft(window))
        return spectrogram

class ParametricSolver:
    """Solver for parametric equations using Fourier methods."""
    
    def __init__(self, n_harmonics: int = 10):
        self.n_harmonics = n_harmonics
        
    def solve_parametric(self, 
                        func: Callable,
                        t_range: Tuple[float, float],
                        n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve parametric equation using Fourier series approximation.
        
        Args:
            func: Function defining the parametric equation
            t_range: Time range (t_start, t_end)
            n_points: Number of points for discretization
            
        Returns:
            Tuple of solution arrays (x(t), y(t))
        """
        t = np.linspace(t_range[0], t_range[1], n_points)
        x_t = np.zeros_like(t)
        y_t = np.zeros_like(t)
        
        # Compute Fourier coefficients
        for n in range(self.n_harmonics):
            an = 2/t_range[1] * integrate.quad(
                lambda t: func(t)[0] * np.cos(2*np.pi*n*t/t_range[1]), 
                0, t_range[1]
            )[0]
            bn = 2/t_range[1] * integrate.quad(
                lambda t: func(t)[1] * np.sin(2*np.pi*n*t/t_range[1]),
                0, t_range[1]
            )[0]
            
            x_t += an * np.cos(2*np.pi*n*t/t_range[1])
            y_t += bn * np.sin(2*np.pi*n*t/t_range[1])
            
        return x_t, y_t

class SpectralDESolver:
    """Differential equation solver using spectral methods."""
    
    def solve_ode(self, 
                 func: Callable,
                 y0: float,
                 t_range: Tuple[float, float],
                 n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using Fourier spectral method.
        
        Args:
            func: ODE function dy/dt = f(t,y)
            y0: Initial condition
            t_range: Time range (t_start, t_end) 
            n_points: Number of discretization points
            
        Returns:
            Tuple of (t, y) arrays
        """
        t = np.linspace(t_range[0], t_range[1], n_points)
        dt = t[1] - t[0]
        
        # Initialize solution in frequency domain
        y_hat = fftpack.fft(np.zeros_like(t))
        y_hat[0] = y0
        
        # Time stepping in frequency domain
        for _ in range(n_points-1):
            # Transform to time domain
            y = fftpack.ifft(y_hat).real
            
            # Evaluate derivative
            dydt = func(t, y)
            
            # Transform back and update
            dydt_hat = fftpack.fft(dydt)
            y_hat = y_hat + dt * dydt_hat
            
        return t, fftpack.ifft(y_hat).real