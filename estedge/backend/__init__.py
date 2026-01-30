"""Backend abstraction for linear algebra."""

from estedge.backend.base import Backend, get_backend
from estedge.backend.jax_backend import JAXBackend
from estedge.backend.numpy_backend import NumPyBackend

__all__ = ["Backend", "NumPyBackend", "JAXBackend", "get_backend"]
