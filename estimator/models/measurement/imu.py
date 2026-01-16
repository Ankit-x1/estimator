"""IMU measurement model."""

from typing import Optional

import numpy as np

from estimator.models.measurement.base import BaseMeasurementModel


class IMU(BaseMeasurementModel):
    """
    IMU measurement model: measures acceleration.

    Assumes state includes acceleration: [p, v, a] for ConstantAcceleration model,
    or works with IMUKinematics where acceleration is computed from control input.

    For IMUKinematics model, IMU is typically used as control input, not measurement.
    This model is designed for ConstantAcceleration process model.
    """

    def __init__(
        self,
        state_dim: int,
        dim: int = 3,
        measurement_noise: float = 0.1,
        acceleration_index: Optional[int] = None,
    ):
        """
        Initialize IMU model.

        Args:
            state_dim: State dimension
            dim: Spatial dimension (1, 2, or 3)
            measurement_noise: Measurement noise standard deviation
            acceleration_index: Starting index of acceleration in state vector.
                              If None, assumes state is [p, v, a] (acceleration at 2*dim)
        """
        if dim not in [1, 2, 3]:
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

        super().__init__(state_dim=state_dim, measurement_dim=dim)
        self.dim = dim
        self.measurement_noise = measurement_noise

        # Determine acceleration index in state
        if acceleration_index is None:
            # Assume state is [p, v, a] for ConstantAcceleration
            if state_dim == 3 * dim:
                self.acceleration_index = 2 * dim
            else:
                raise ValueError(
                    f"Cannot determine acceleration index. "
                    f"For state_dim={state_dim} and dim={dim}, "
                    f"expected state_dim=3*dim={3*dim} for [p, v, a] state"
                )
        else:
            self.acceleration_index = acceleration_index

        # Validate acceleration index
        if self.acceleration_index + dim > state_dim:
            raise ValueError(
                f"Acceleration index {self.acceleration_index} + dim {dim} "
                f"exceeds state_dim {state_dim}"
            )

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement prediction: returns acceleration from state.

        Args:
            x: State vector [p, v, a] or custom format

        Returns:
            Predicted acceleration measurement (dim,)
        """
        return x[self.acceleration_index : self.acceleration_index + self.dim].copy()

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian: H = dh/dx

        Returns:
            Jacobian matrix (dim, state_dim) where H[i, acceleration_index+i] = 1
        """
        H = np.zeros((self.measurement_dim, self.state_dim))
        for i in range(self.dim):
            H[i, self.acceleration_index + i] = 1.0
        return H

    def covariance(self) -> np.ndarray:
        """Measurement noise covariance"""
        return np.eye(self.measurement_dim) * (self.measurement_noise**2)
