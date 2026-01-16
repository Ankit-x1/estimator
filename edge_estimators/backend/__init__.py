"""Backend abstraction for linear algebra."""

from edge_estimators.backend.base import Backend, get_backend
from edge_estimators.backend.jax_backend import JAXBackend
from edge_estimators.backend.numpy_backend import NumPyBackend

__all__ = ["Backend", "NumPyBackend", "JAXBackend", "get_backend"]
