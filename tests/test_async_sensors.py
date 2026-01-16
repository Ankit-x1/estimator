"""Test asynchronous sensor fusion."""

import numpy as np

from edge_estimators.core.estimator import EKF
from edge_estimators.core.state import State
from edge_estimators.models.measurement.encoder import Encoder
from edge_estimators.models.measurement.gps import GPS
from edge_estimators.models.process.constant_velocity import ConstantVelocity


def test_async_sensors():
    """Test fusion of sensors at different rates."""
    # IMU-like fast sensor (encoder) at 10Hz, GPS at 1Hz
    initial_x = np.array([0.0, 0.0, 0.0, 0.0])  # [px, py, vx, vy]
    initial_P = np.eye(4) * 10.0
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    process_model = ConstantVelocity(dim=2, process_noise=0.01)
    encoder_model = Encoder(state_dim=4, measure_position=True, dim=2, measurement_noise=0.1)
    gps_model = GPS(state_dim=4, dim=2, measurement_noise=1.0)

    ekf = EKF(process_model, {"encoder": encoder_model, "gps": gps_model}, initial_state)

    dt_fast = 0.1  # 10Hz
    # dt_slow = 1.0  # 1Hz (calculated from update frequency)

    true_velocity = np.array([1.0, 0.5])
    timestamp = 0.0

    # Simulate 5 seconds
    for step in range(50):
        timestamp += dt_fast

        # Always predict with fast rate
        ekf.predict(u=np.array([]), dt=dt_fast, timestamp=timestamp)

        # Fast sensor (encoder) every step
        true_pos = true_velocity * timestamp
        encoder_measurement = true_pos + np.random.normal(0, 0.1, size=2)
        ekf.update(z=encoder_measurement, sensor_name="encoder", timestamp=timestamp)

        # Slow sensor (GPS) every 10 steps
        if step % 10 == 0:
            gps_measurement = true_pos + np.random.normal(0, 1.0, size=2)
            ekf.update(z=gps_measurement, sensor_name="gps", timestamp=timestamp)

    # Should track velocity reasonably well
    estimated_velocity = ekf.state.x[2:4]
    error = np.linalg.norm(estimated_velocity - true_velocity)
    assert error < 0.5, f"Velocity error too large: {error}"
