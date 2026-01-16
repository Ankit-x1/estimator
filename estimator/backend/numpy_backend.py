"""NumPy backend for production deployment."""

import numpy as np

from estimator.backend.base import Backend


class NumPyBackend(Backend):
    """NumPy-based backend for linear algebra."""

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication."""
        return A @ B

    def inv(self, A: np.ndarray) -> np.ndarray:
        """Matrix inverse."""
        return np.linalg.inv(A)

    def cholesky(self, A: np.ndarray) -> np.ndarray:
        """Cholesky decomposition."""
        return np.linalg.cholesky(A)

    def eye(self, n: int) -> np.ndarray:
        """Identity matrix."""
        return np.eye(n)

    def zeros(self, shape: tuple) -> np.ndarray:
        """Zero matrix."""
        return np.zeros(shape)

    def array(self, data) -> np.ndarray:
        """Convert to array."""
        return np.asarray(data, dtype=np.float64)

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system."""
        return np.linalg.solve(A, b)
