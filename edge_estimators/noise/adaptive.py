"""Adaptive noise estimation: innovation-based adaptive Q/R."""

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np


class AdaptiveNoise:
    """
    Innovation-based adaptive noise estimation.

    Adapts Q and R based on innovation statistics.
    Supports per-sensor R matrices for multi-sensor systems.
    """

    def __init__(
        self,
        Q_init: np.ndarray,
        R_init: Optional[np.ndarray] = None,
        R_init_dict: Optional[Dict[str, np.ndarray]] = None,
        window_size: int = 50,
        adaptation_rate: float = 0.1,
    ):
        """
        Initialize adaptive noise model.

        Args:
            Q_init: Initial process noise covariance
            R_init: Initial measurement noise covariance (single sensor)
            R_init_dict: Initial measurement noise covariance per sensor (multi-sensor)
            window_size: Window size for innovation statistics
            adaptation_rate: Rate of adaptation (0-1)

        Note: Either R_init or R_init_dict should be provided, not both.
        """
        self.Q = Q_init.copy()
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate

        # Support both single-sensor and multi-sensor R
        if R_init_dict is not None:
            self.R_dict = {k: v.copy() for k, v in R_init_dict.items()}
            self.single_sensor = False
        elif R_init is not None:
            self.R = R_init.copy()
            self.R_dict = {}
            self.single_sensor = True
        else:
            raise ValueError("Either R_init or R_init_dict must be provided")

        # Innovation history (per sensor for multi-sensor)
        self.innovations: Deque[np.ndarray] = deque(maxlen=window_size)
        self.innovation_covariances: Deque[np.ndarray] = deque(maxlen=window_size)
        self.innovations_dict: Dict[str, Deque[np.ndarray]] = {}
        self.innovation_covariances_dict: Dict[str, Deque[np.ndarray]] = {}

    def add_innovation(
        self, innovation: np.ndarray, S: np.ndarray, sensor_name: Optional[str] = None
    ):
        """
        Add innovation to history.

        Args:
            innovation: Innovation vector (y - h(x))
            S: Innovation covariance (H*P*H^T + R)
            sensor_name: Sensor name (for multi-sensor systems)
        """
        if self.single_sensor or sensor_name is None:
            self.innovations.append(innovation.copy())
            self.innovation_covariances.append(S.copy())
        else:
            if sensor_name not in self.innovations_dict:
                self.innovations_dict[sensor_name] = deque(maxlen=self.window_size)
                self.innovation_covariances_dict[sensor_name] = deque(maxlen=self.window_size)
            self.innovations_dict[sensor_name].append(innovation.copy())
            self.innovation_covariances_dict[sensor_name].append(S.copy())

    def adapt_R(self, sensor_name: Optional[str] = None):
        """
        Adapt measurement noise R based on innovation statistics.

        Uses: R_new = (1-α)*R_old + α*R_estimated

        Args:
            sensor_name: Sensor name (for multi-sensor systems)
        """
        if self.single_sensor or sensor_name is None:
            if len(self.innovations) < 10:
                return  # Not enough data

            # Compute sample covariance of innovations
            innovations_array = np.array(list(self.innovations))
            S_array = np.array(list(self.innovation_covariances))

            # Expected innovation covariance: E[ν*ν^T] = S
            # If innovations are larger than expected, increase R
            if innovations_array.shape[0] > 1:
                innovation_cov = np.cov(innovations_array.T)
            else:
                innovation_cov = np.outer(innovations_array[0], innovations_array[0])

            expected_cov = np.mean(S_array, axis=0)

            # Adapt R: if actual > expected, increase R
            R_estimated = innovation_cov - expected_cov + self.R
            R_estimated = np.maximum(R_estimated, 0.01 * self.R)  # Lower bound

            # Update with adaptation rate
            self.R = (1 - self.adaptation_rate) * self.R + self.adaptation_rate * R_estimated
        else:
            if sensor_name not in self.innovations_dict:
                return
            if len(self.innovations_dict[sensor_name]) < 10:
                return  # Not enough data

            innovations_array = np.array(list(self.innovations_dict[sensor_name]))
            S_array = np.array(list(self.innovation_covariances_dict[sensor_name]))

            if innovations_array.shape[0] > 1:
                innovation_cov = np.cov(innovations_array.T)
            else:
                innovation_cov = np.outer(innovations_array[0], innovations_array[0])

            expected_cov = np.mean(S_array, axis=0)
            R_current = self.R_dict[sensor_name]

            R_estimated = innovation_cov - expected_cov + R_current
            R_estimated = np.maximum(R_estimated, 0.01 * R_current)  # Lower bound

            self.R_dict[sensor_name] = (
                1 - self.adaptation_rate
            ) * R_current + self.adaptation_rate * R_estimated

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

    def get_R(self, sensor_name: Optional[str] = None) -> np.ndarray:
        """
        Get measurement noise (adapts over time).

        Args:
            sensor_name: Sensor name (for multi-sensor systems)

        Returns:
            Measurement noise covariance R
        """
        if self.single_sensor or sensor_name is None:
            return self.R
        else:
            if sensor_name not in self.R_dict:
                raise ValueError(f"Unknown sensor: {sensor_name}")
            return self.R_dict[sensor_name]
