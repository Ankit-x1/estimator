"""Adaptive noise estimation: innovation-based adaptive Q/R."""

import numpy as np
from collections import deque
from typing import Optional, Deque


class AdaptiveNoise:
    """
    Innovation-based adaptive noise estimation.
    
    Adapts Q and R based on innovation statistics.
    """
    
    def __init__(
        self,
        Q_init: np.ndarray,
        R_init: np.ndarray,
        window_size: int = 50,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize adaptive noise model.
        
        Args:
            Q_init: Initial process noise covariance
            R_init: Initial measurement noise covariance
            window_size: Window size for innovation statistics
            adaptation_rate: Rate of adaptation (0-1)
        """
        self.Q = Q_init.copy()
        self.R = R_init.copy()
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        # Innovation history
        self.innovations: Deque[np.ndarray] = deque(maxlen=window_size)
        self.innovation_covariances: Deque[np.ndarray] = deque(maxlen=window_size)
    
    def add_innovation(self, innovation: np.ndarray, S: np.ndarray):
        """
        Add innovation to history.
        
        Args:
            innovation: Innovation vector (y - h(x))
            S: Innovation covariance (H*P*H^T + R)
        """
        self.innovations.append(innovation.copy())
        self.innovation_covariances.append(S.copy())
    
    def adapt_R(self):
        """
        Adapt measurement noise R based on innovation statistics.
        
        Uses: R_new = (1-α)*R_old + α*R_estimated
        """
        if len(self.innovations) < 10:
            return  # Not enough data
        
        # Compute sample covariance of innovations
        innovations_array = np.array(list(self.innovations))
        S_array = np.array(list(self.innovation_covariances))
        
        # Expected innovation covariance: E[ν*ν^T] = S
        # If innovations are larger than expected, increase R
        innovation_cov = np.cov(innovations_array.T)
        expected_cov = np.mean(S_array, axis=0)
        
        # Adapt R: if actual > expected, increase R
        R_estimated = innovation_cov - expected_cov + self.R
        R_estimated = np.maximum(R_estimated, 0.01 * self.R)  # Lower bound
        
        # Update with adaptation rate
        self.R = (1 - self.adaptation_rate) * self.R + self.adaptation_rate * R_estimated
    
    def adapt_Q(self):
        """
        Adapt process noise Q based on innovation statistics.
        
        Similar to adapt_R but for process noise.
        """
        if len(self.innovations) < 10:
            return  # Not enough data
        
        # For Q adaptation, we typically need state prediction errors
        # Simplified: increase Q if innovations are consistently large
        innovations_array = np.array(list(self.innovations))
        innovation_magnitude = np.mean(np.linalg.norm(innovations_array, axis=1))
        
        # If innovations are large, increase Q
        if innovation_magnitude > 1.0:
            scale = 1.0 + self.adaptation_rate * (innovation_magnitude - 1.0)
            self.Q = self.Q * scale
    
    def get_Q(self, dt: float) -> np.ndarray:
        """Get process noise (adapts over time)."""
        return self.Q
    
    def get_R(self) -> np.ndarray:
        """Get measurement noise (adapts over time)."""
        return self.R

