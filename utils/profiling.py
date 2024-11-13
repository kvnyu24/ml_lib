"""Profiling and performance measurement utilities."""

import time
import sys
import cProfile
import pstats
from typing import Any, Callable
from functools import wraps
import logging
import psutil
import numpy as np

logger = logging.getLogger(__name__)

def time_function(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def memory_usage(obj: Any) -> float:
    """Get memory usage of object in MB."""
    return sys.getsizeof(obj) / (1024 * 1024)

def profile_code(func: Callable) -> Callable:
    """Decorator to profile code execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats()
        return result
    return wrapper

def get_object_size(obj: Any) -> float:
    """Get total size of object and all its attributes in MB."""
    seen = set()
    def sizeof(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(obj)
        if hasattr(obj, '__dict__'):
            size += sizeof(obj.__dict__)
        elif isinstance(obj, dict):
            size += sum(sizeof(k) + sizeof(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(sizeof(i) for i in obj)
        return size
    return sizeof(obj) / (1024 * 1024)