"""Custom exceptions for the library."""

class MLLibraryError(Exception):
    """Base exception class for all library errors."""
    pass

class NotFittedError(MLLibraryError):
    """Exception raised when predicting with an unfitted model."""
    pass

class ValidationError(MLLibraryError):
    """Exception raised for invalid input data."""
    pass

class ConvergenceError(MLLibraryError):
    """Exception raised when optimization fails to converge."""
    pass

class DimensionalityError(MLLibraryError):
    """Exception raised for incompatible dimensions."""
    pass

class ParameterError(MLLibraryError):
    """Exception raised for invalid parameter values."""
    pass

class ConfigurationError(MLLibraryError):
    """Exception raised for invalid configuration."""
    pass