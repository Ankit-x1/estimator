"""IMU kinematics process model."""

import numpy as np

from estimator.models.process.base import BaseProcessModel


class IMUKinematics(BaseProcessModel):
    """
    IMU kinematics model: v += (a_meas - bias) * dt; p += v*dt; b_dot = 0

    State: [p, v, bias] (position, velocity, acceleration bias)
    Control: [a_meas] (measured acceleration)
    """

    def __init__(self, dim: int = 3, process_noise: float = 0.01, bias_noise: float = 1e-6):
        """
        Initialize IMU kinematics model.

        Args:
            dim: Spatial dimension (1, 2, or 3)
            process_noise: Process noise for position/velocity
            bias_noise: Process noise for bias (typically very small)
        """
        if dim not in [1, 2, 3]:
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

        super().__init__(state_dim=3 * dim)
        self.dim = dim
        self.process_noise = process_noise
        self.bias_noise = bias_noise

    def f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        State propagation: v += (a_meas - bias) * dt; p += v*dt; bias unchanged

        Args:
            x: State [p, v, bias] (3*dim,)
            u: Control [a_meas] (dim,)
            dt: Time step
        """
        x_next = x.copy()
        a_meas = u[: self.dim] if len(u) >= self.dim else np.zeros(self.dim)
        bias = x[2 * self.dim : 3 * self.dim]

        # v += (a_meas - bias) * dt
        for i in range(self.dim):
            x_next[self.dim + i] += (a_meas[i] - bias[i]) * dt

        # p += v*dt
        for i in range(self.dim):
            x_next[i] += x_next[self.dim + i] * dt

        # bias unchanged (b_dot = 0)
        return x_next

    def jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """State transition Jacobian"""
        n = self.state_dim
        F = np.eye(n)

        for i in range(self.dim):
            # dp/dv = dt
            F[i, self.dim + i] = dt
            # dv/dbias = -dt
            F[self.dim + i, 2 * self.dim + i] = -dt

        return F

    def noise(self, dt: float) -> np.ndarray:
        """Process noise covariance"""
        n = self.state_dim
        Q = np.zeros((n, n))

        # Position and velocity noise
        for i in range(2 * self.dim):
            Q[i, i] = (self.process_noise**2) * dt

        # Bias noise (very small, random walk)
        for i in range(2 * self.dim, n):
            Q[i, i] = (self.bias_noise**2) * dt

        return Q
