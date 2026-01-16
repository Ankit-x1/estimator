"""Estimator orchestration: KF, EKF, UKF."""

import numpy as np
from typing import Optional, Dict, Any, List
from edge_estimators.core.state import State
from edge_estimators.core.time import compute_dt, clamp_dt
from edge_estimators.core.covariance import update_covariance
from edge_estimators.core.exceptions import EstimatorError
from edge_estimators.utils.checks import check_state, check_measurement
from edge_estimators.utils.linalg import enforce_psd, symmetrize
from edge_estimators.backend.base import Backend
from edge_estimators.backend.numpy_backend import NumPyBackend
from edge_estimators.models.process.base import BaseProcessModel
from edge_estimators.models.measurement.base import BaseMeasurementModel
from edge_estimators.noise.static import StaticNoise
from edge_estimators.noise.adaptive import AdaptiveNoise
from edge_estimators.gating.mahalanobis import MahalanobisGate


class KF:
    """
    Linear Kalman Filter.
    
    For linear systems with linear measurements.
    """
    
    def __init__(
        self,
        process_model: BaseProcessModel,
        measurement_model: BaseMeasurementModel,
        initial_state: State,
        backend: Optional[Backend] = None,
        noise_model: Optional[StaticNoise] = None,
        gate: Optional[MahalanobisGate] = None
    ):
        """
        Initialize linear Kalman filter.
        
        Args:
            process_model: Process model
            measurement_model: Measurement model
            initial_state: Initial state
            backend: Backend for linear algebra (default: NumPy)
            noise_model: Noise model (default: from models)
            gate: Outlier rejection gate (default: disabled)
        """
        self.process_model = process_model
        self.measurement_model = measurement_model
        self.state = initial_state.clone()
        self.backend = backend or NumPyBackend()
        self.gate = gate or MahalanobisGate(enabled=False)
        
        # Validate dimensions
        if process_model.state_dim != initial_state.n:
            raise ValueError(
                f"Process model state_dim ({process_model.state_dim}) != "
                f"initial state dim ({initial_state.n})"
            )
        
        # Noise model
        if noise_model is None:
            # Create default from models
            Q = process_model.noise(0.01)  # Default dt
            R = measurement_model.covariance()
            self.noise_model = StaticNoise(Q=Q, R=R)
        else:
            self.noise_model = noise_model
    
    def predict(self, u: np.ndarray, dt: float) -> State:
        """
        Prediction step: x = F*x + B*u, P = F*P*F^T + Q
        
        Args:
            u: Control input
            dt: Time step
        
        Returns:
            Updated state
        """
        dt = clamp_dt(dt)
        
        # State prediction: x = F*x (linear)
        F = self.process_model.jacobian(self.state.x, u, dt)
        x_pred = self.process_model.f(self.state.x, u, dt)
        
        # Covariance prediction: P = F*P*F^T + Q
        Q = self.noise_model.get_Q(dt)
        P_pred = F @ self.state.P @ F.T + Q
        
        # Enforce symmetry and PSD
        P_pred = symmetrize(P_pred)
        P_pred = enforce_psd(P_pred)
        
        # Update state
        self.state.x = x_pred
        self.state.P = P_pred
        
        # Update timestamp
        self.state.timestamp += dt
        
        # Check for errors
        check_state(self.state)
        
        return self.state
    
    def update(self, z: np.ndarray, timestamp: Optional[float] = None) -> State:
        """
        Update step: K = P*H^T*(H*P*H^T + R)^-1, x = x + K*(z - h(x)), P = (I - K*H)*P
        
        Args:
            z: Measurement
            timestamp: Measurement timestamp (optional)
        
        Returns:
            Updated state
        """
        check_measurement(z, "measurement")
        
        # Measurement prediction
        h_x = self.measurement_model.h(self.state.x)
        innovation = z - h_x
        
        # Measurement Jacobian
        H = self.measurement_model.jacobian(self.state.x)
        R = self.measurement_model.covariance()
        
        # Innovation covariance: S = H*P*H^T + R
        S = H @ self.state.P @ H.T + R
        S = symmetrize(S)
        S = enforce_psd(S)
        
        # Gating: reject outliers
        if not self.gate.check(innovation, S):
            return self.state  # Reject measurement
        
        # Kalman gain: K = P*H^T * S^-1
        K = self.state.P @ H.T @ self.backend.inv(S)
        
        # State update: x = x + K*innovation
        self.state.x = self.state.x + K @ innovation
        
        # Covariance update: P = (I - K*H)*P (Joseph form)
        self.state.P = update_covariance(
            self.state.P, K, H, R, use_joseph=True, enforce_psd_flag=True
        )
        
        # Update timestamp if provided
        if timestamp is not None:
            self.state.timestamp = timestamp
        
        # Check for errors
        check_state(self.state)
        
        return self.state


class EKF:
    """
    Extended Kalman Filter.
    
    For nonlinear systems with nonlinear measurements.
    """
    
    def __init__(
        self,
        process_model: BaseProcessModel,
        measurement_models: Dict[str, BaseMeasurementModel],
        initial_state: State,
        backend: Optional[Backend] = None,
        noise_model: Optional[StaticNoise] = None,
        adaptive_noise: Optional[AdaptiveNoise] = None,
        gate: Optional[MahalanobisGate] = None
    ):
        """
        Initialize Extended Kalman filter.
        
        Args:
            process_model: Process model
            measurement_models: Dictionary of measurement models (keyed by sensor name)
            initial_state: Initial state
            backend: Backend for linear algebra
            noise_model: Static noise model
            adaptive_noise: Adaptive noise model (optional)
            gate: Outlier rejection gate
        """
        self.process_model = process_model
        self.measurement_models = measurement_models
        self.state = initial_state.clone()
        self.backend = backend or NumPyBackend()
        self.gate = gate or MahalanobisGate(enabled=True)
        self.adaptive_noise = adaptive_noise
        
        # Validate dimensions
        if process_model.state_dim != initial_state.n:
            raise ValueError(
                f"Process model state_dim ({process_model.state_dim}) != "
                f"initial state dim ({initial_state.n})"
            )
        
        # Noise model
        if noise_model is None:
            # Create default from models
            Q = process_model.noise(0.01)
            # Use first measurement model for default R
            first_model = list(measurement_models.values())[0]
            R = first_model.covariance()
            self.noise_model = StaticNoise(Q=Q, R=R)
        else:
            self.noise_model = noise_model
    
    def predict(self, u: np.ndarray, dt: float, timestamp: Optional[float] = None) -> State:
        """
        Prediction step: x = f(x, u, dt), P = F*P*F^T + Q
        
        Args:
            u: Control input
            dt: Time step
            timestamp: Prediction timestamp (optional)
        
        Returns:
            Updated state
        """
        dt = clamp_dt(dt)
        
        # State prediction: x = f(x, u, dt) (nonlinear)
        x_pred = self.process_model.f(self.state.x, u, dt)
        
        # Covariance prediction: P = F*P*F^T + Q
        F = self.process_model.jacobian(self.state.x, u, dt)
        Q = self.noise_model.get_Q(dt)
        if self.adaptive_noise:
            Q = self.adaptive_noise.get_Q(dt)
        
        P_pred = F @ self.state.P @ F.T + Q
        
        # Enforce symmetry and PSD
        P_pred = symmetrize(P_pred)
        P_pred = enforce_psd(P_pred)
        
        # Update state
        self.state.x = x_pred
        self.state.P = P_pred
        
        # Update timestamp
        if timestamp is not None:
            self.state.timestamp = timestamp
        else:
            self.state.timestamp += dt
        
        # Check for errors
        check_state(self.state)
        
        return self.state
    
    def update(
        self,
        z: np.ndarray,
        sensor_name: str,
        timestamp: Optional[float] = None
    ) -> State:
        """
        Update step for a specific sensor.
        
        Args:
            z: Measurement
            sensor_name: Name of sensor (key in measurement_models)
            timestamp: Measurement timestamp (optional)
        
        Returns:
            Updated state
        """
        if sensor_name not in self.measurement_models:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        
        measurement_model = self.measurement_models[sensor_name]
        check_measurement(z, f"sensor {sensor_name}")
        
        # Measurement prediction: h(x)
        h_x = measurement_model.h(self.state.x)
        innovation = z - h_x
        
        # Measurement Jacobian: H = dh/dx
        H = measurement_model.jacobian(self.state.x)
        R = measurement_model.covariance()
        if self.adaptive_noise:
            R = self.adaptive_noise.get_R()
        
        # Innovation covariance: S = H*P*H^T + R
        S = H @ self.state.P @ H.T + R
        S = symmetrize(S)
        S = enforce_psd(S)
        
        # Gating: reject outliers
        if not self.gate.check(innovation, S):
            return self.state  # Reject measurement
        
        # Kalman gain: K = P*H^T * S^-1
        K = self.state.P @ H.T @ self.backend.inv(S)
        
        # State update: x = x + K*innovation
        self.state.x = self.state.x + K @ innovation
        
        # Covariance update: P = (I - K*H)*P (Joseph form)
        self.state.P = update_covariance(
            self.state.P, K, H, R, use_joseph=True, enforce_psd_flag=True
        )
        
        # Adaptive noise update
        if self.adaptive_noise:
            self.adaptive_noise.add_innovation(innovation, S)
            self.adaptive_noise.adapt_R()
            self.adaptive_noise.adapt_Q()
        
        # Update timestamp if provided
        if timestamp is not None:
            self.state.timestamp = timestamp
        
        # Check for errors
        check_state(self.state)
        
        return self.state


class UKF:
    """
    Unscented Kalman Filter.
    
    Uses sigma points for nonlinear state estimation.
    """
    
    def __init__(
        self,
        process_model: BaseProcessModel,
        measurement_models: Dict[str, BaseMeasurementModel],
        initial_state: State,
        backend: Optional[Backend] = None,
        noise_model: Optional[StaticNoise] = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        gate: Optional[MahalanobisGate] = None
    ):
        """
        Initialize Unscented Kalman filter.
        
        Args:
            process_model: Process model
            measurement_models: Dictionary of measurement models
            initial_state: Initial state
            backend: Backend for linear algebra
            noise_model: Noise model
            alpha: Spread parameter (default: 1e-3)
            beta: Prior knowledge parameter (default: 2.0 for Gaussian)
            kappa: Secondary scaling parameter (default: 0.0)
            gate: Outlier rejection gate
        """
        self.process_model = process_model
        self.measurement_models = measurement_models
        self.state = initial_state.clone()
        self.backend = backend or NumPyBackend()
        self.gate = gate or MahalanobisGate(enabled=True)
        
        self.n = initial_state.n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # UKF parameters
        self.lambda_ = alpha ** 2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        
        # Weights
        W0_m = self.lambda_ / (self.n + self.lambda_)
        W0_c = W0_m + (1 - alpha ** 2 + beta)
        Wi = 1 / (2 * (self.n + self.lambda_))
        
        self.weights_m = np.array([W0_m] + [Wi] * (2 * self.n))
        self.weights_c = np.array([W0_c] + [Wi] * (2 * self.n))
        
        # Noise model
        if noise_model is None:
            Q = process_model.noise(0.01)
            first_model = list(measurement_models.values())[0]
            R = first_model.covariance()
            self.noise_model = StaticNoise(Q=Q, R=R)
        else:
            self.noise_model = noise_model
    
    def _compute_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute sigma points using Merwe's method.
        
        Args:
            x: Mean state
            P: Covariance
        
        Returns:
            Sigma points (2*n+1, n)
        """
        # Cholesky decomposition
        try:
            L = self.backend.cholesky(P)
        except np.linalg.LinAlgError:
            # If P is not positive definite, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(P)
            eigenvals = np.maximum(eigenvals, 1e-8)
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Generate sigma points
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = x
        for i in range(self.n):
            sigma_points[i + 1] = x + self.gamma * L[i]
            sigma_points[i + 1 + self.n] = x - self.gamma * L[i]
        
        return sigma_points
    
    def predict(self, u: np.ndarray, dt: float, timestamp: Optional[float] = None) -> State:
        """
        Prediction step using sigma points.
        
        Args:
            u: Control input
            dt: Time step
            timestamp: Prediction timestamp (optional)
        
        Returns:
            Updated state
        """
        dt = clamp_dt(dt)
        
        # Generate sigma points
        sigma_points = self._compute_sigma_points(self.state.x, self.state.P)
        
        # Propagate sigma points through process model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * self.n + 1):
            sigma_points_pred[i] = self.process_model.f(sigma_points[i], u, dt)
        
        # Compute predicted mean
        x_pred = np.sum(self.weights_m[:, None] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance
        P_pred = np.zeros((self.n, self.n))
        Q = self.noise_model.get_Q(dt)
        for i in range(2 * self.n + 1):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.weights_c[i] * np.outer(diff, diff)
        P_pred += Q
        
        # Enforce symmetry and PSD
        P_pred = symmetrize(P_pred)
        P_pred = enforce_psd(P_pred)
        
        # Update state
        self.state.x = x_pred
        self.state.P = P_pred
        
        # Update timestamp
        if timestamp is not None:
            self.state.timestamp = timestamp
        else:
            self.state.timestamp += dt
        
        # Check for errors
        check_state(self.state)
        
        return self.state
    
    def update(
        self,
        z: np.ndarray,
        sensor_name: str,
        timestamp: Optional[float] = None
    ) -> State:
        """
        Update step using sigma points.
        
        Args:
            z: Measurement
            sensor_name: Name of sensor
            timestamp: Measurement timestamp (optional)
        
        Returns:
            Updated state
        """
        if sensor_name not in self.measurement_models:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        
        measurement_model = self.measurement_models[sensor_name]
        check_measurement(z, f"sensor {sensor_name}")
        
        # Generate sigma points
        sigma_points = self._compute_sigma_points(self.state.x, self.state.P)
        
        # Propagate through measurement model
        m = measurement_model.measurement_dim
        z_sigma = np.zeros((2 * self.n + 1, m))
        for i in range(2 * self.n + 1):
            z_sigma[i] = measurement_model.h(sigma_points[i])
        
        # Compute predicted measurement mean
        z_pred = np.sum(self.weights_m[:, None] * z_sigma, axis=0)
        
        # Innovation
        innovation = z - z_pred
        
        # Innovation covariance
        S = np.zeros((m, m))
        R = measurement_model.covariance()
        for i in range(2 * self.n + 1):
            diff = z_sigma[i] - z_pred
            S += self.weights_c[i] * np.outer(diff, diff)
        S += R
        S = symmetrize(S)
        S = enforce_psd(S)
        
        # Cross covariance
        Pxz = np.zeros((self.n, m))
        for i in range(2 * self.n + 1):
            diff_x = sigma_points[i] - self.state.x
            diff_z = z_sigma[i] - z_pred
            Pxz += self.weights_c[i] * np.outer(diff_x, diff_z)
        
        # Gating
        if not self.gate.check(innovation, S):
            return self.state
        
        # Kalman gain
        K = Pxz @ self.backend.inv(S)
        
        # State update
        self.state.x = self.state.x + K @ innovation
        
        # Covariance update
        self.state.P = self.state.P - K @ S @ K.T
        self.state.P = symmetrize(self.state.P)
        self.state.P = enforce_psd(self.state.P)
        
        # Update timestamp
        if timestamp is not None:
            self.state.timestamp = timestamp
        
        # Check for errors
        check_state(self.state)
        
        return self.state

