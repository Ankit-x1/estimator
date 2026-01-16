"""Abstract base class for backend."""

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Abstract backend for linear algebra operations."""

    @abstractmethod
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication: A @ B."""
        pass

    @abstractmethod
    def inv(self, A: np.ndarray) -> np.ndarray:
        """Matrix inverse: A^-1."""
        pass

    @abstractmethod
    def cholesky(self, A: np.ndarray) -> np.ndarray:
        """Cholesky decomposition: A = L @ L^T."""
        pass

    @abstractmethod
    def eye(self, n: int) -> np.ndarray:
        """Identity matrix of size n."""
        pass

    @abstractmethod
    def zeros(self, shape: tuple) -> np.ndarray:
        """Zero matrix of given shape."""
        pass

    @abstractmethod
    def array(self, data) -> np.ndarray:
        """Convert to array."""
        pass

    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system: A @ x = b."""
        pass


def get_backend(name: str = "numpy") -> Backend:
    """
    Get backend by name.

    Args:
        name: Backend name ("numpy" or "jax")

    Returns:
        Backend instance
    """
    if name == "numpy":
        from estimator.backend.numpy_backend import NumPyBackend

        return NumPyBackend()
    elif name == "jax":
        try:
            from estimator.backend.jax_backend import JAXBackend

            return JAXBackend()
        except ImportError:
            raise ImportError("JAX not installed. Install with: pip install jax jaxlib")
    else:
        raise ValueError(f"Unknown backend: {name}")
