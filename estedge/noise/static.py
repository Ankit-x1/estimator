"""Static noise models: fixed Q/R matrices."""

from typing import Optional

import numpy as np


class StaticNoise:
    """Static noise model with fixed Q and R matrices."""

    def __init__(self, Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None):
        """
        Initialize static noise model.

        Args:
            Q: Process noise covariance (fixed)
            R: Measurement noise covariance (fixed)
        """
        self.Q = Q
        self.R = R

    def get_Q(self, dt: float) -> np.ndarray:
        """
        Get process noise for time step.

        Args:
            dt: Time step

        Returns:
            Process noise matrix Q
        """
        if self.Q is None:
            raise ValueError("Q not set")
        return self.Q

    def get_R(self) -> np.ndarray:
        """
        Get measurement noise.

        Returns:
            Measurement noise matrix R
        """
        if self.R is None:
            raise ValueError("R not set")
        return self.R

    def update_Q(self, Q: np.ndarray):
        """Update process noise."""
        self.Q = Q

    def update_R(self, R: np.ndarray):
        """Update measurement noise."""
        self.R = R
