"""NaN, divergence, stability checks."""

import numpy as np

from edge_estimators.core.exceptions import DivergenceError, NaNError, PSDViolationError
from edge_estimators.utils.linalg import is_psd


def check_state(state, max_trace: float = 1e6) -> None:
    """
    Check state for NaN, Inf, and divergence.

    Args:
        state: State object with x and P
        max_trace: Maximum allowed trace(P) before divergence (default: 1e6)

    Raises:
        NaNError if NaN found
        DivergenceError if covariance explodes
        PSDViolationError if P is not PSD
    """
    # Check for NaN
    if np.any(np.isnan(state.x)) or np.any(np.isnan(state.P)):
        raise NaNError("NaN detected in state")

    # Check for Inf
    if np.any(np.isinf(state.x)) or np.any(np.isinf(state.P)):
        raise DivergenceError("Inf detected in state")

    # Check covariance trace (divergence)
    trace_P = np.trace(state.P)
    if trace_P > max_trace:
        raise DivergenceError(f"Covariance trace too large: {trace_P:.2e} > {max_trace:.2e}")

    # Check PSD
    if not is_psd(state.P):
        raise PSDViolationError("Covariance is not positive semi-definite")


def check_measurement(z: np.ndarray, name: str = "measurement") -> None:
    """
    Check measurement for validity.

    Args:
        z: Measurement vector
        name: Name for error message

    Raises:
        ValueError if invalid
    """
    z = np.asarray(z)
    if np.any(np.isnan(z)):
        raise ValueError(f"NaN in {name}")
    if np.any(np.isinf(z)):
        raise ValueError(f"Inf in {name}")
