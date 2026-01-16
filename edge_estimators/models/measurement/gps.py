"""GPS measurement model."""

import numpy as np

from edge_estimators.models.measurement.base import BaseMeasurementModel


class GPS(BaseMeasurementModel):
    """
    GPS measurement model: measures position only.

    Low-rate, noisy position measurements.
    """

    def __init__(self, state_dim: int, dim: int = 3, measurement_noise: float = 1.0):
        """
        Initialize GPS model.

        Args:
            state_dim: State dimension
            dim: Spatial dimension (2 for lat/lon, 3 for lat/lon/alt)
            measurement_noise: Measurement noise standard deviation (meters)
        """
        if dim not in [2, 3]:
            raise ValueError(f"dim must be 2 or 3, got {dim}")

        super().__init__(state_dim=state_dim, measurement_dim=dim)
        self.dim = dim
        self.measurement_noise = measurement_noise

    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement prediction: position"""
        return x[: self.dim].copy()

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Measurement Jacobian"""
        H = np.zeros((self.measurement_dim, self.state_dim))
        for i in range(self.dim):
            H[i, i] = 1.0
        return H

    def covariance(self) -> np.ndarray:
        """Measurement noise covariance"""
        return np.eye(self.measurement_dim) * (self.measurement_noise**2)
