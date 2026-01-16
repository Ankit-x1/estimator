"""Optional: record state & covariance over time."""

from typing import List, Dict, Any
import numpy as np
from edge_estimators.core.state import State


class StateLogger:
    """Logger for recording estimator state over time."""
    
    def __init__(self):
        """Initialize empty logger."""
        self.timestamps: List[float] = []
        self.states: List[np.ndarray] = []
        self.covariances: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
    
    def log(self, state: State):
        """
        Log a state snapshot.
        
        Args:
            state: State to log
        """
        self.timestamps.append(state.timestamp)
        self.states.append(state.x.copy())
        self.covariances.append(state.P.copy())
        self.metadata.append(state.metadata.copy())
    
    def clear(self):
        """Clear all logged data."""
        self.timestamps.clear()
        self.states.clear()
        self.covariances.clear()
        self.metadata.clear()
    
    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Get logged trajectory as arrays.
        
        Returns:
            Dictionary with 'timestamps', 'states', 'covariances'
        """
        return {
            "timestamps": np.array(self.timestamps),
            "states": np.array(self.states),
            "covariances": np.array(self.covariances),
            "metadata": self.metadata
        }

