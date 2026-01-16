"""Encoder measurement model."""

import numpy as np
from edge_estimators.models.measurement.base import BaseMeasurementModel


class Encoder(BaseMeasurementModel):
    """
    Encoder measurement model: measures position or velocity.
    
    Can measure position (1D/2D/3D) or velocity.
    """
    
    def __init__(
        self,
        state_dim: int,
        measure_position: bool = True,
        dim: int = 1,
        measurement_noise: float = 0.01
    ):
        """
        Initialize encoder model.
        
        Args:
            state_dim: State dimension
            measure_position: If True, measure position; else measure velocity
            dim: Spatial dimension (1, 2, or 3)
            measurement_noise: Measurement noise standard deviation
        """
        if dim not in [1, 2, 3]:
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
        
        super().__init__(state_dim=state_dim, measurement_dim=dim)
        self.measure_position = measure_position
        self.dim = dim
        self.measurement_noise = measurement_noise
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement prediction: position or velocity"""
        if self.measure_position:
            return x[:self.dim].copy()
        else:
            # Assume velocity is at indices [dim:2*dim]
            return x[self.dim:2*self.dim].copy()
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Measurement Jacobian"""
        H = np.zeros((self.measurement_dim, self.state_dim))
        if self.measure_position:
            for i in range(self.dim):
                H[i, i] = 1.0
        else:
            for i in range(self.dim):
                H[i, self.dim + i] = 1.0
        return H
    
    def covariance(self) -> np.ndarray:
        """Measurement noise covariance"""
        return np.eye(self.measurement_dim) * (self.measurement_noise ** 2)

