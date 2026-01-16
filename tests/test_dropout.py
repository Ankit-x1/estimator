"""Test filter recovery from measurement dropout."""

import numpy as np

from edge_estimators.core.estimator import KF
from edge_estimators.core.state import State
from edge_estimators.models.measurement.encoder import Encoder
from edge_estimators.models.process.constant_velocity import ConstantVelocity


def test_dropout_recovery():
    """Test that filter recovers after missing measurements."""
    initial_x = np.array([0.0, 1.0])  # position=0, velocity=1
    initial_P = np.eye(2) * 0.1
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    process_model = ConstantVelocity(dim=1, process_noise=0.01)
    measurement_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

    kf = KF(process_model, measurement_model, initial_state)

    dt = 0.1
    true_velocity = 1.0

    # Normal operation
    for i in range(20):
        kf.predict(u=np.array([]), dt=dt)
        true_pos = true_velocity * (i + 1) * dt
        kf.update(z=np.array([true_pos + np.random.normal(0, 0.1)]))

    # Dropout: 100ms without measurements (10 steps)
    for i in range(10):
        kf.predict(u=np.array([]), dt=dt)
        # No update

    # Recovery: measurements resume
    for i in range(20):
        kf.predict(u=np.array([]), dt=dt)
        true_pos = true_velocity * (30 + i + 1) * dt
        kf.update(z=np.array([true_pos + np.random.normal(0, 0.1)]))

    # Filter should still track velocity
    estimated_velocity = kf.state.x[1]
    assert abs(estimated_velocity - true_velocity) < 0.3, "Filter did not recover from dropout"

    # Covariance should increase during dropout, then decrease
    assert np.trace(kf.state.P) < 10.0, "Covariance should be bounded"
