"""Magnetometer measurement model."""

import numpy as np
from edge_estimators.models.measurement.base import BaseMeasurementModel


class Magnetometer(BaseMeasurementModel):
    """
    Magnetometer measurement model: measures heading/yaw.
    
    Optional heading measurement.
    """
    
    def __init__(
        self,
        state_dim: int,
        measurement_noise: float = 0.1
    ):
        """
        Initialize magnetometer model.
        
        Args:
            state_dim: State dimension
            measurement_noise: Measurement noise standard deviation (radians)
        """
        super().__init__(state_dim=state_dim, measurement_dim=1)
        self.measurement_noise = measurement_noise
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement prediction: heading.
        
        Assumes heading is at index 2 (for 2D) or index 5 (for 3D yaw).
        This is a simplified model.
        """
        # Placeholder: assumes heading is in state
        # In practice, would extract from quaternion or Euler angles
        if self.state_dim >= 3:
            # Assume heading is at some index (simplified)
            return np.array([0.0])  # Placeholder
        else:
            return np.array([0.0])
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Measurement Jacobian"""
        H = np.zeros((1, self.state_dim))
        # Placeholder: depends on state representation
        return H
    
    def covariance(self) -> np.ndarray:
        """Measurement noise covariance"""
        return np.array([[self.measurement_noise ** 2]])

