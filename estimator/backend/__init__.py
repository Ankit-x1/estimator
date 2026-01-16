"""Backend abstraction for linear algebra."""

from estimator.backend.base import Backend, get_backend
from estimator.backend.jax_backend import JAXBackend
from estimator.backend.numpy_backend import NumPyBackend

__all__ = ["Backend", "NumPyBackend", "JAXBackend", "get_backend"]
