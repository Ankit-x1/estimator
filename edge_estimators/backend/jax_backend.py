"""JAX backend for research and auto-jacobian support."""

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from edge_estimators.backend.base import Backend

if TYPE_CHECKING:
    import jax.numpy as jnp
    from jax import jit
else:
    try:
        import jax.numpy as jnp
        from jax import jit

        JAX_AVAILABLE = True
    except ImportError:
        JAX_AVAILABLE = False
        jnp = None  # type: ignore
        jit = None  # type: ignore


class JAXBackend(Backend):
    """JAX-based backend with optional JIT compilation."""

    def __init__(self, use_jit: bool = False):
        """
        Initialize JAX backend.

        Args:
            use_jit: Enable JIT compilation (default: False)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX not installed. Install with: pip install jax jaxlib")

        self.use_jit = use_jit
        if jnp is None or jit is None:
            raise ImportError("JAX not installed. Install with: pip install jax jaxlib")
        self._matmul: Callable[[Any, Any], Any] = jit(jnp.matmul) if use_jit else jnp.matmul
        self._inv: Callable[[Any], Any] = jit(jnp.linalg.inv) if use_jit else jnp.linalg.inv
        self._cholesky: Callable[[Any], Any] = (
            jit(jnp.linalg.cholesky) if use_jit else jnp.linalg.cholesky
        )
        self._solve: Callable[[Any, Any], Any] = (
            jit(jnp.linalg.solve) if use_jit else jnp.linalg.solve
        )

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication."""
        result = self._matmul(A, B)
        return np.asarray(result)

    def inv(self, A: np.ndarray) -> np.ndarray:
        """Matrix inverse."""
        result = self._inv(A)
        return np.asarray(result)

    def cholesky(self, A: np.ndarray) -> np.ndarray:
        """Cholesky decomposition."""
        result = self._cholesky(A)
        return np.asarray(result)

    def eye(self, n: int) -> np.ndarray:
        """Identity matrix."""
        return np.asarray(jnp.eye(n))

    def zeros(self, shape: tuple) -> np.ndarray:
        """Zero matrix."""
        return np.asarray(jnp.zeros(shape))

    def array(self, data) -> np.ndarray:
        """Convert to array."""
        return np.asarray(jnp.array(data), dtype=np.float64)

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system."""
        result = self._solve(A, b)
        return np.asarray(result)
