"""State container: holds x, P, timestamp, metadata."""

import copy
from typing import Any, Dict, Optional

import numpy as np


class State:
    """State container for estimator state and covariance."""

    def __init__(
        self,
        x: np.ndarray,
        P: np.ndarray,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize state container.

        Args:
            x: State vector (n,)
            P: Covariance matrix (n, n)
            timestamp: Current timestamp in seconds
            metadata: Optional metadata dictionary
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.P = np.asarray(P, dtype=np.float64)
        self.timestamp = float(timestamp)
        self.metadata = metadata or {}

        # Validate dimensions
        if self.x.ndim != 1:
            raise ValueError(f"State x must be 1D, got shape {self.x.shape}")
        if self.P.ndim != 2:
            raise ValueError(f"Covariance P must be 2D, got shape {self.P.shape}")
        if self.x.shape[0] != self.P.shape[0] or self.x.shape[0] != self.P.shape[1]:
            raise ValueError(f"Dimension mismatch: x.shape={self.x.shape}, P.shape={self.P.shape}")

    def clone(self) -> "State":
        """Create a deep copy of the state."""
        return State(
            x=self.x.copy(),
            P=self.P.copy(),
            timestamp=self.timestamp,
            metadata=copy.deepcopy(self.metadata),
        )

    def reset(self, x: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None):
        """Reset state and/or covariance."""
        if x is not None:
            self.x = np.asarray(x, dtype=np.float64)
        if P is not None:
            self.P = np.asarray(P, dtype=np.float64)

    @property
    def n(self) -> int:
        """State dimension."""
        return self.x.shape[0]

    def __repr__(self) -> str:
        return (
            f"State(n={self.n}, timestamp={self.timestamp:.3f}, "
            f"trace(P)={np.trace(self.P):.6f})"
        )
