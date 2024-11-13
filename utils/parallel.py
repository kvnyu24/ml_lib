"""Parallel processing utilities for efficient computation."""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, List, Any, Optional, Iterator, Dict
import numpy as np
from functools import partial
from tqdm import tqdm
from core import get_logger

logger = get_logger(__name__)

class ThreadPool:
    """Thread-based parallel executor for I/O-bound tasks."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize thread pool.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers or (2 * mp.cpu_count())
        
    def map(self, func: Callable, iterable: Iterator,
            show_progress: bool = False,
            **kwargs) -> List[Any]:
        """Execute function in parallel using threads.
        
        Args:
            func: Function to execute
            iterable: Input iterator
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments to pass to function
            
        Returns:
            List of results
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if kwargs:
                func = partial(func, **kwargs)
                
            futures = list(executor.map(func, iterable))
            
            if show_progress:
                futures = list(tqdm(futures, total=len(iterable)))
                
            return futures

class ProcessPool:
    """Process-based parallel executor for CPU-bound tasks."""
    
    def __init__(self, max_workers: Optional[int] = None,
                 maxtasksperchild: Optional[int] = None):
        """Initialize process pool.
        
        Args:
            max_workers: Maximum number of worker processes
            maxtasksperchild: Maximum tasks per child process
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.maxtasksperchild = maxtasksperchild
        
    def map(self, func: Callable, iterable: Iterator,
            show_progress: bool = False,
            chunksize: Optional[int] = None,
            **kwargs) -> List[Any]:
        """Execute function in parallel using processes.
        
        Args:
            func: Function to execute
            iterable: Input iterator
            show_progress: Whether to show progress bar
            chunksize: Size of input chunks per process
            **kwargs: Additional arguments to pass to function
            
        Returns:
            List of results
        """
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context('spawn')
        ) as executor:
            if kwargs:
                func = partial(func, **kwargs)
                
            futures = list(executor.map(func, iterable, chunksize=chunksize))
            
            if show_progress:
                futures = list(tqdm(futures, total=len(iterable)))
                
            return futures

def parallel_map(func: Callable,
                iterable: Iterator,
                n_jobs: int = -1,
                backend: str = 'processes',
                show_progress: bool = False,
                **kwargs) -> List[Any]:
    """High-level parallel map function.
    
    Args:
        func: Function to execute
        iterable: Input iterator
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        backend: 'processes' or 'threads'
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments to pass to function
        
    Returns:
        List of results
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
        
    if backend == 'threads':
        pool = ThreadPool(max_workers=n_jobs)
    elif backend == 'processes':
        pool = ProcessPool(max_workers=n_jobs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
        
    return pool.map(func, iterable, show_progress=show_progress, **kwargs)

def parallel_apply(func: Callable,
                  data: np.ndarray,
                  axis: int = 0,
                  n_jobs: int = -1,
                  show_progress: bool = False,
                  **kwargs) -> np.ndarray:
    """Apply function to array in parallel along an axis.
    
    Args:
        func: Function to apply
        data: Input array
        axis: Axis along which to apply function
        n_jobs: Number of parallel jobs
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments to pass to function
        
    Returns:
        Array of results
    """
    results = parallel_map(
        func,
        (x for x in np.rollaxis(data, axis)),
        n_jobs=n_jobs,
        show_progress=show_progress,
        **kwargs
    )
    return np.stack(results, axis=axis)

def distribute_tasks(tasks: List[Any],
                    n_workers: int = -1) -> List[List[Any]]:
    """Distribute tasks among workers.
    
    Args:
        tasks: List of tasks to distribute
        n_workers: Number of workers (-1 for CPU count)
        
    Returns:
        List of task chunks for each worker
    """
    if n_workers == -1:
        n_workers = mp.cpu_count()
        
    n_tasks = len(tasks)
    chunk_size = n_tasks // n_workers
    remainder = n_tasks % n_workers
    
    chunks = []
    start = 0
    
    for i in range(n_workers):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(tasks[start:start + size])
        start += size
        
    return chunks

class TaskQueue:
    """Queue for parallel task processing."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize task queue.
        
        Args:
            max_workers: Maximum number of worker processes
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.queue = mp.Queue()
        self.results = mp.Queue()
        self.workers = []
        
    def worker(self, func: Callable):
        """Worker process function."""
        while True:
            try:
                task = self.queue.get()
                if task is None:
                    break
                result = func(task)
                self.results.put(result)
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                self.results.put(None)
                
    def map(self, func: Callable, tasks: List[Any]) -> List[Any]:
        """Process tasks in parallel.
        
        Args:
            func: Function to process tasks
            tasks: List of tasks
            
        Returns:
            List of results
        """
        # Start workers
        for _ in range(self.max_workers):
            p = mp.Process(target=self.worker, args=(func,))
            p.start()
            self.workers.append(p)
            
        # Add tasks to queue
        for task in tasks:
            self.queue.put(task)
            
        # Add termination signals
        for _ in range(self.max_workers):
            self.queue.put(None)
            
        # Collect results
        results = []
        for _ in range(len(tasks)):
            results.append(self.results.get())
            
        # Wait for workers
        for p in self.workers:
            p.join()
            
        return results 