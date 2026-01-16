"""
Example: IMU + Encoder simulation.

Demonstrates basic sensor fusion with synthetic sensors.
"""

import matplotlib.pyplot as plt
import numpy as np

from estimator.core.estimator import EKF
from estimator.core.state import State
from estimator.models.measurement.encoder import Encoder
from estimator.models.measurement.gps import GPS
from estimator.models.process.constant_velocity import ConstantVelocity


def simulate_imu_encoder():
    """Simulate IMU and encoder sensors."""
    # Initial state: [px, py, vx, vy]
    initial_x = np.array([0.0, 0.0, 1.0, 0.5])
    initial_P = np.eye(4) * 10.0
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    # Models
    process_model = ConstantVelocity(dim=2, process_noise=0.01)
    encoder_model = Encoder(state_dim=4, measure_position=True, dim=2, measurement_noise=0.1)
    gps_model = GPS(state_dim=4, dim=2, measurement_noise=1.0)

    # Filter
    ekf = EKF(process_model, {"encoder": encoder_model, "gps": gps_model}, initial_state)

    # Simulation parameters
    dt = 0.1
    duration = 10.0
    steps = int(duration / dt)

    # True trajectory
    true_velocity = np.array([1.0, 0.5])
    true_positions = []
    estimated_positions = []
    timestamps = []

    # Simulate
    for i in range(steps):
        t = i * dt

        # Predict
        ekf.predict(u=np.array([]), dt=dt, timestamp=t)

        # True position
        true_pos = true_velocity * t
        true_positions.append(true_pos.copy())

        # Encoder measurement (high rate, low noise)
        if i % 1 == 0:  # Every step
            encoder_z = true_pos + np.random.normal(0, 0.1, size=2)
            ekf.update(z=encoder_z, sensor_name="encoder", timestamp=t)

        # GPS measurement (low rate, high noise)
        if i % 10 == 0:  # Every 10 steps
            gps_z = true_pos + np.random.normal(0, 1.0, size=2)
            ekf.update(z=gps_z, sensor_name="gps", timestamp=t)

        estimated_positions.append(ekf.state.x[:2].copy())
        timestamps.append(t)

    # Plot results
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(true_positions[:, 0], true_positions[:, 1], "b-", label="True", linewidth=2)
    plt.plot(
        estimated_positions[:, 0], estimated_positions[:, 1], "r--", label="Estimated", linewidth=2
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    error = np.linalg.norm(estimated_positions - true_positions, axis=1)
    plt.plot(timestamps, error, "g-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title("Estimation Error")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("imu_encoder_sim.png", dpi=150)
    print("Saved plot to imu_encoder_sim.png")
    print(f"Final position error: {error[-1]:.3f} m")
    print(f"Final velocity estimate: {ekf.state.x[2:4]}")
    print(f"True velocity: {true_velocity}")


if __name__ == "__main__":
    simulate_imu_encoder()
