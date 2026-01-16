"""Test numerical stability: PSD, NaN handling."""

import numpy as np

from estimator.core.estimator import KF
from estimator.core.state import State
from estimator.models.measurement.encoder import Encoder
from estimator.models.process.constant_velocity import ConstantVelocity
from estimator.utils.linalg import is_psd


def test_covariance_psd():
    """Test that covariance remains PSD."""
    initial_x = np.array([0.0, 1.0])
    initial_P = np.eye(2) * 0.1
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    process_model = ConstantVelocity(dim=1, process_noise=0.01)
    measurement_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

    kf = KF(process_model, measurement_model, initial_state)

    dt = 0.1
    for i in range(100):
        kf.predict(u=np.array([]), dt=dt)
        kf.update(z=np.array([i * dt + np.random.normal(0, 0.1)]))

        # Check PSD
        assert is_psd(kf.state.P), f"Covariance not PSD at step {i}"

        # Check no NaN
        assert not np.any(np.isnan(kf.state.x)), f"NaN in state at step {i}"
        assert not np.any(np.isnan(kf.state.P)), f"NaN in covariance at step {i}"


def test_small_dt():
    """Test handling of very small time steps."""
    initial_x = np.array([0.0, 1.0])
    initial_P = np.eye(2) * 0.1
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    process_model = ConstantVelocity(dim=1, process_noise=0.01)
    measurement_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

    kf = KF(process_model, measurement_model, initial_state)

    # Very small dt
    dt = 1e-6
    kf.predict(u=np.array([]), dt=dt)
    kf.update(z=np.array([0.0]))

    # Should not crash
    assert is_psd(kf.state.P), "Covariance should remain PSD"


def test_large_dt():
    """Test handling of large time steps."""
    initial_x = np.array([0.0, 1.0])
    initial_P = np.eye(2) * 0.1
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    process_model = ConstantVelocity(dim=1, process_noise=0.01)
    measurement_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

    kf = KF(process_model, measurement_model, initial_state)

    # Large dt (should be clamped)
    dt = 10.0
    kf.predict(u=np.array([]), dt=dt)
    kf.update(z=np.array([0.0]))

    # Should not crash
    assert is_psd(kf.state.P), "Covariance should remain PSD"
