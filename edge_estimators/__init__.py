"""
edge-estimators: Production-focused state estimation runtime.

Sensor-first, deployment-ready, numerically robust.
"""

from edge_estimators.core.estimator import KF, EKF, UKF
from edge_estimators.core.state import State

__version__ = "0.1.0"
__all__ = ["KF", "EKF", "UKF", "State"]

