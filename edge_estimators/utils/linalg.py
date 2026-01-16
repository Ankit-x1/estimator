"""Linear algebra utilities: PSD clamp, symmetrize, matrix checks."""

import numpy as np


def symmetrize(A: np.ndarray) -> np.ndarray:
    """
    Enforce symmetry: A = (A + A^T) / 2

    Args:
        A: Matrix to symmetrize

    Returns:
        Symmetric matrix
    """
    return (A + A.T) / 2


def enforce_psd(A: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """
    Enforce positive semi-definiteness by clamping eigenvalues.

    Args:
        A: Matrix to make PSD
        min_eigenvalue: Minimum allowed eigenvalue (default: 1e-8)

    Returns:
        PSD matrix
    """
    A = symmetrize(A)
    eigenvals, eigenvecs = np.linalg.eigh(A)

    # Clamp negative eigenvalues
    eigenvals = np.maximum(eigenvals, min_eigenvalue)

    # Reconstruct matrix
    A_psd = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    return A_psd


def is_psd(A: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if matrix is positive semi-definite.

    Args:
        A: Matrix to check
        tol: Tolerance for eigenvalue check

    Returns:
        True if PSD
    """
    eigenvals = np.linalg.eigvalsh(A)
    return np.all(eigenvals >= -tol)


def check_nan(A: np.ndarray, name: str = "matrix") -> None:
    """
    Check for NaN values and raise if found.

    Args:
        A: Matrix to check
        name: Name for error message

    Raises:
        ValueError if NaN found
    """
    if np.any(np.isnan(A)):
        raise ValueError(f"NaN detected in {name}")


def check_inf(A: np.ndarray, name: str = "matrix") -> None:
    """
    Check for Inf values and raise if found.

    Args:
        A: Matrix to check
        name: Name for error message

    Raises:
        ValueError if Inf found
    """
    if np.any(np.isinf(A)):
        raise ValueError(f"Inf detected in {name}")
