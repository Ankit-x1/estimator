"""Magnetometer measurement model."""

from typing import Optional

import numpy as np

from estimator.models.measurement.base import BaseMeasurementModel


class Magnetometer(BaseMeasurementModel):
    """
    Magnetometer measurement model: measures heading/yaw.

    Measures heading angle (yaw) from state. Assumes heading is stored directly
    in the state vector at a specified index.

    Note: For full 3D orientation, you would need quaternions or Euler angles
    in the state. This is a simplified model for 2D/planar navigation.
    """

    def __init__(
        self, state_dim: int, measurement_noise: float = 0.1, heading_index: Optional[int] = None
    ):
        """
        Initialize magnetometer model.

        Args:
            state_dim: State dimension
            measurement_noise: Measurement noise standard deviation (radians)
            heading_index: Index of heading in state vector. If None, attempts
                          to infer: for 2D state [x, y, vx, vy, heading], heading_index=4
                          For 3D state [x, y, z, vx, vy, vz, heading], heading_index=6
        """
        super().__init__(state_dim=state_dim, measurement_dim=1)
        self.measurement_noise = measurement_noise

        # Determine heading index
        if heading_index is None:
            # Try to infer: common patterns
            # 2D: [x, y, vx, vy, heading] -> index 4
            # 3D: [x, y, z, vx, vy, vz, heading] -> index 6
            # Or: [x, y, heading] -> index 2
            if state_dim == 3:
                self.heading_index = 2  # [x, y, heading]
            elif state_dim == 5:
                self.heading_index = 4  # [x, y, vx, vy, heading]
            elif state_dim == 7:
                self.heading_index = 6  # [x, y, z, vx, vy, vz, heading]
            else:
                raise ValueError(
                    f"Cannot infer heading_index for state_dim={state_dim}. "
                    f"Please specify heading_index explicitly."
                )
        else:
            self.heading_index = heading_index

        # Validate heading index
        if self.heading_index >= state_dim:
            raise ValueError(f"heading_index {self.heading_index} >= state_dim {state_dim}")

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement prediction: returns heading angle.

        Args:
            x: State vector containing heading at heading_index

        Returns:
            Predicted heading measurement [heading] (1,)
        """
        return np.array([x[self.heading_index]])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian: H = dh/dx

        Returns:
            Jacobian matrix (1, state_dim) where H[0, heading_index] = 1
        """
        H = np.zeros((1, self.state_dim))
        H[0, self.heading_index] = 1.0
        return H

    def covariance(self) -> np.ndarray:
        """Measurement noise covariance"""
        return np.array([[self.measurement_noise**2]])
