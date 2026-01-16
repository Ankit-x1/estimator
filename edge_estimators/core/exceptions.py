"""Custom exceptions for estimator errors."""


class EstimatorError(Exception):
    """Base exception for estimator errors."""

    pass


class NaNError(EstimatorError):
    """Raised when NaN detected in state or covariance."""

    pass


class PSDViolationError(EstimatorError):
    """Raised when covariance is not positive semi-definite."""

    pass


class DivergenceError(EstimatorError):
    """Raised when filter diverges (covariance explodes)."""

    pass


class SensorError(EstimatorError):
    """Raised when sensor measurement is invalid."""

    pass
