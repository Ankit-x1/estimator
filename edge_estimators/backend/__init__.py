"""Backend abstraction for linear algebra."""

from edge_estimators.backend.base import Backend
from edge_estimators.backend.numpy_backend import NumPyBackend
from edge_estimators.backend.jax_backend import JAXBackend

__all__ = ["Backend", "NumPyBackend", "JAXBackend"]

