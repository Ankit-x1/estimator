"""Covariance updates: Joseph form, PSD enforcement."""

import numpy as np

from estimator.utils.linalg import enforce_psd, symmetrize


def joseph_update(P: np.ndarray, K: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Joseph form covariance update for numerical stability.

    P = (I - K*H) * P * (I - K*H)^T + K * R * K^T

    Args:
        P: Prior covariance (n, n)
        K: Kalman gain (n, m)
        H: Measurement Jacobian (m, n)
        R: Measurement noise (m, m)

    Returns:
        Updated covariance (n, n)
    """
    I = np.eye(P.shape[0])
    IKH = I - K @ H
    P_updated = IKH @ P @ IKH.T + K @ R @ K.T
    return P_updated


def update_covariance(
    P: np.ndarray,
    K: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    use_joseph: bool = True,
    enforce_psd_flag: bool = True,
) -> np.ndarray:
    """
    Update covariance with optional Joseph form and PSD enforcement.

    Args:
        P: Prior covariance
        K: Kalman gain
        H: Measurement Jacobian
        R: Measurement noise
        use_joseph: Use Joseph form (default: True)
        enforce_psd_flag: Enforce PSD (default: True)

    Returns:
        Updated covariance
    """
    if use_joseph:
        P_new = joseph_update(P, K, H, R)
    else:
        # Standard update: P = (I - K*H) * P
        I = np.eye(P.shape[0])
        P_new = (I - K @ H) @ P

    # Enforce symmetry and PSD
    P_new = symmetrize(P_new)
    if enforce_psd_flag:
        P_new = enforce_psd(P_new)

    return P_new
