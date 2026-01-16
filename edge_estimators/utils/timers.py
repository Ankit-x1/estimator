"""Benchmarks, dt logging."""

import time
import numpy as np
from typing import Dict, List
from collections import defaultdict


class Timer:
    """Simple timer for benchmarking."""
    
    def __init__(self):
        """Initialize timer."""
        self.times: Dict[str, List[float]] = defaultdict(list)
    
    def start(self, name: str) -> float:
        """
        Start timing an operation.
        
        Args:
            name: Operation name
        
        Returns:
            Start time
        """
        return time.perf_counter()
    
    def end(self, name: str, start_time: float):
        """
        End timing and record duration.
        
        Args:
            name: Operation name
            start_time: Start time from start()
        """
        duration = time.perf_counter() - start_time
        self.times[name].append(duration)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for an operation.
        
        Args:
            name: Operation name
        
        Returns:
            Dictionary with 'mean', 'std', 'min', 'max', 'count'
        """
        if name not in self.times or len(self.times[name]) == 0:
            return {}
        
        times = self.times[name]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "count": len(times)
        }
    
    def clear(self):
        """Clear all timing data."""
        self.times.clear()

