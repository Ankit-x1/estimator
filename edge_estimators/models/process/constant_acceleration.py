"""Constant acceleration process model."""

import numpy as np

from edge_estimators.models.process.base import BaseProcessModel


class ConstantAcceleration(BaseProcessModel):
    """
    Constant acceleration model: v += a*dt; p += v*dt

    State: [p, v, a] (position, velocity, acceleration)
    """

    def __init__(self, dim: int = 1, process_noise: float = 0.01):
        """
        Initialize constant acceleration model.

        Args:
            dim: Spatial dimension (1, 2, or 3)
            process_noise: Process noise standard deviation
        """
        if dim not in [1, 2, 3]:
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

        super().__init__(state_dim=3 * dim)
        self.dim = dim
        self.process_noise = process_noise

    def f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """State propagation: v += a*dt; p += v*dt"""
        x_next = x.copy()
        for i in range(self.dim):
            # v += a*dt
            x_next[self.dim + i] += x[2 * self.dim + i] * dt
            # p += v*dt
            x_next[i] += x_next[self.dim + i] * dt
        return x_next

    def jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """State transition Jacobian"""
        n = self.state_dim
        F = np.eye(n)
        for i in range(self.dim):
            F[i, self.dim + i] = dt  # dp/dv = dt
            F[self.dim + i, 2 * self.dim + i] = dt  # dv/da = dt
        return F

    def noise(self, dt: float) -> np.ndarray:
        """Process noise covariance"""
        n = self.state_dim
        Q = np.eye(n) * (self.process_noise**2) * dt
        return Q
