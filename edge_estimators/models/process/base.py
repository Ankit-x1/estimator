"""Abstract base class for process models."""

from abc import ABC, abstractmethod
import numpy as np


class BaseProcessModel(ABC):
    """Abstract base class for process models."""
    
    def __init__(self, state_dim: int):
        """
        Initialize process model.
        
        Args:
            state_dim: Dimension of state vector
        """
        self.state_dim = state_dim
    
    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        State propagation: x_{k+1} = f(x_k, u_k, dt)
        
        Args:
            x: Current state (n,)
            u: Control input (m,)
            dt: Time step
        
        Returns:
            Next state (n,)
        """
        pass
    
    @abstractmethod
    def jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition Jacobian: F = df/dx
        
        Args:
            x: Current state
            u: Control input
            dt: Time step
        
        Returns:
            Jacobian matrix (n, n)
        """
        pass
    
    @abstractmethod
    def noise(self, dt: float) -> np.ndarray:
        """
        Process noise covariance: Q(dt)
        
        Args:
            dt: Time step
        
        Returns:
            Process noise matrix (n, n)
        """
        pass

