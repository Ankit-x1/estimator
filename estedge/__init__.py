"""
estedge: Production-focused state estimation runtime.

Sensor-first, deployment-ready, numerically robust.
"""

from estedge.core.estimator import EKF, KF, UKF
from estedge.core.state import State

__version__ = "0.1.0"
__all__ = ["KF", "EKF", "UKF", "State"]
