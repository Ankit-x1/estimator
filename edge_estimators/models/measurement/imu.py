"""IMU measurement model."""

import numpy as np
from edge_estimators.models.measurement.base import BaseMeasurementModel


class IMU(BaseMeasurementModel):
    """
    IMU measurement model: measures acceleration and/or angular rate.
    
    For simplicity, assumes acceleration measurement.
    """
    
    def __init__(
        self,
        state_dim: int,
        dim: int = 3,
        measurement_noise: float = 0.1,
        measure_acceleration: bool = True
    ):
        """
        Initialize IMU model.
        
        Args:
            state_dim: State dimension
            dim: Spatial dimension (1, 2, or 3)
            measurement_noise: Measurement noise standard deviation
            measure_acceleration: If True, measure acceleration; else angular rate
        """
        if dim not in [1, 2, 3]:
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
        
        super().__init__(state_dim=state_dim, measurement_dim=dim)
        self.dim = dim
        self.measurement_noise = measurement_noise
        self.measure_acceleration = measure_acceleration
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement prediction.
        
        If measuring acceleration, returns acceleration from state.
        For IMU kinematics model, acceleration = (a_meas - bias) from control.
        Here we assume acceleration is directly in state or computed.
        """
        # For simplicity, assume we measure acceleration directly
        # In practice, this would depend on state representation
        if self.measure_acceleration:
            # If state has acceleration, return it; else return zeros
            if self.state_dim >= 3 * self.dim:
                # State: [p, v, a] or [p, v, bias]
                return np.zeros(self.dim)  # Placeholder
            else:
                return np.zeros(self.dim)
        else:
            return np.zeros(self.dim)
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Measurement Jacobian"""
        H = np.zeros((self.measurement_dim, self.state_dim))
        # Placeholder: depends on state representation
        return H
    
    def covariance(self) -> np.ndarray:
        """Measurement noise covariance"""
        return np.eye(self.measurement_dim) * (self.measurement_noise ** 2)

