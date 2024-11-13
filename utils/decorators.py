"""Utility decorators for function and class behavior modification."""

import time
import logging
import functools
import warnings
from typing import Callable, Optional, Any
from core.exceptions import NotFittedError

logger = logging.getLogger(__name__)

def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.4f} seconds to execute")
        return result
    return wrapper

def cache_result(maxsize: Optional[int] = None) -> Callable:
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
                if maxsize and len(cache) > maxsize:
                    # Remove oldest entry
                    cache.pop(next(iter(cache)))
            return cache[key]
        return wrapper
    return decorator

def log_calls(level: str = 'INFO') -> Callable:
    """Decorator to log function calls."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_msg = f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            getattr(logger, level.lower())(log_msg)
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        return wrapper
    return decorator

def deprecate(message: Optional[str] = None) -> Callable:
    """Decorator to mark functions as deprecated."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_msg = message or f"{func.__name__} is deprecated and will be removed in a future version."
            warnings.warn(warn_msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_fitted(attribute: str = 'is_fitted_') -> Callable:
    """Decorator to check if estimator is fitted before calling method."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, attribute):
                raise NotFittedError(
                    f"This {type(self).__name__} instance is not fitted yet. "
                    f"Call 'fit' with appropriate arguments before using this method."
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """Decorator to retry function execution on failure."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    logger.warning(
                        f"Attempt {attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
        return wrapper
    return decorator

def memoize(func: Callable) -> Callable:
    """Decorator to memoize function results."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper 