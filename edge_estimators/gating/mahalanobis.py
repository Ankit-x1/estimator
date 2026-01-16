"""Mahalanobis gating for outlier rejection."""

import numpy as np


class MahalanobisGate:
    """
    Mahalanobis distance gating for outlier rejection.

    Rejects measurements if: innovation^T * S^-1 * innovation > threshold
    """

    def __init__(self, threshold: float = 9.0, enabled: bool = True):
        """
        Initialize Mahalanobis gate.

        Args:
            threshold: Chi-squared threshold (default: 9.0 for 3-sigma)
            enabled: Enable/disable gating
        """
        self.threshold = threshold
        self.enabled = enabled
        self.rejected_count = 0
        self.accepted_count = 0

    def check(self, innovation: np.ndarray, S: np.ndarray) -> bool:
        """
        Check if measurement passes gate.

        Args:
            innovation: Innovation vector (y - h(x))
            S: Innovation covariance (H*P*H^T + R)

        Returns:
            True if measurement passes gate (should be used)
        """
        if not self.enabled:
            return True

        try:
            # Mahalanobis distance: d^2 = innovation^T * S^-1 * innovation
            S_inv = np.linalg.inv(S)
            mahalanobis_sq = innovation.T @ S_inv @ innovation

            if mahalanobis_sq <= self.threshold:
                self.accepted_count += 1
                return True
            else:
                self.rejected_count += 1
                return False
        except np.linalg.LinAlgError:
            # If S is singular, reject measurement
            self.rejected_count += 1
            return False

    def get_stats(self) -> dict:
        """Get gating statistics."""
        total = self.accepted_count + self.rejected_count
        if total == 0:
            return {"accepted": 0, "rejected": 0, "rejection_rate": 0.0}

        return {
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "rejection_rate": self.rejected_count / total,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.rejected_count = 0
        self.accepted_count = 0
