"""Spectral models package

This package contains spectral analysis and Fourier-based models including:
- Fourier series models with adaptive frequency selection
- Spectral differential equation solvers
"""

from .fourier import (
    BaseFourierModel,
    AdaptiveFourierModel,
    SpectralKernel,
    TimeFrequencyAnalyzer,
    SpectralDESolver
)

__all__ = [
    'BaseFourierModel',
    'AdaptiveFourierModel', 
    'SpectralKernel',
    'TimeFrequencyAnalyzer',
    'SpectralDESolver'
]
