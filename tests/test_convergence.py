"""Test filter convergence under noise."""

import numpy as np

from estimator.core.estimator import EKF, KF
from estimator.core.state import State
from estimator.models.measurement.encoder import Encoder
from estimator.models.process.constant_velocity import ConstantVelocity


def test_kf_convergence():
    """Test that KF converges to true state under noise."""
    # True state: position=10, velocity=1
    true_state = np.array([10.0, 1.0])

    # Initial estimate: position=0, velocity=0 (uncertain)
    initial_x = np.array([0.0, 0.0])
    initial_P = np.eye(2) * 100.0  # High uncertainty
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    # Models
    process_model = ConstantVelocity(dim=1, process_noise=0.01)
    measurement_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

    # Filter
    kf = KF(process_model, measurement_model, initial_state)

    # Simulate with noisy measurements
    dt = 0.1
    for i in range(100):
        # Predict
        kf.predict(u=np.array([]), dt=dt)

        # True position with noise
        true_pos = true_state[0] + true_state[1] * (i + 1) * dt
        noisy_measurement = true_pos + np.random.normal(0, 0.1)

        # Update
        kf.update(z=np.array([noisy_measurement]))

    # Check convergence: estimate should be close to true state
    estimated_velocity = kf.state.x[1]
    assert abs(estimated_velocity - true_state[1]) < 0.2, "Velocity not converged"

    # Covariance should decrease
    assert np.trace(kf.state.P) < np.trace(initial_P), "Covariance should decrease"


def test_ekf_convergence():
    """Test that EKF converges for nonlinear system."""
    # Similar to KF but with EKF
    true_state = np.array([10.0, 1.0])

    initial_x = np.array([0.0, 0.0])
    initial_P = np.eye(2) * 100.0
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    process_model = ConstantVelocity(dim=1, process_noise=0.01)
    measurement_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

    ekf = EKF(process_model, {"encoder": measurement_model}, initial_state)

    dt = 0.1
    for i in range(100):
        ekf.predict(u=np.array([]), dt=dt)

        true_pos = true_state[0] + true_state[1] * (i + 1) * dt
        noisy_measurement = true_pos + np.random.normal(0, 0.1)

        ekf.update(z=np.array([noisy_measurement]), sensor_name="encoder")

    estimated_velocity = ekf.state.x[1]
    assert abs(estimated_velocity - true_state[1]) < 0.2, "EKF velocity not converged"
