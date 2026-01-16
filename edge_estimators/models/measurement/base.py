"""Abstract base class for measurement models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseMeasurementModel(ABC):
    """Abstract base class for measurement models."""

    def __init__(self, state_dim: int, measurement_dim: int):
        """
        Initialize measurement model.

        Args:
            state_dim: Dimension of state vector
            measurement_dim: Dimension of measurement vector
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

    @abstractmethod
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement prediction: z = h(x)

        Args:
            x: Current state (n,)

        Returns:
            Predicted measurement (m,)
        """
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement Jacobian: H = dh/dx

        Args:
            x: Current state

        Returns:
            Jacobian matrix (m, n)
        """
        pass

    @abstractmethod
    def covariance(self) -> np.ndarray:
        """
        Measurement noise covariance: R

        Returns:
            Measurement noise matrix (m, m)
        """
        pass
