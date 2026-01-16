"""
Example: Asynchronous sensor fusion.

Demonstrates handling sensors at different rates (IMU 400Hz, GPS 5Hz).
"""

import matplotlib.pyplot as plt
import numpy as np

from edge_estimators.core.estimator import EKF
from edge_estimators.core.state import State
from edge_estimators.models.measurement.encoder import Encoder
from edge_estimators.models.measurement.gps import GPS
from edge_estimators.models.process.constant_velocity import ConstantVelocity


def simulate_async_sensors():
    """Simulate asynchronous sensors at different rates."""
    # Initial state: [px, py, vx, vy]
    initial_x = np.array([0.0, 0.0, 2.0, 1.0])
    initial_P = np.eye(4) * 5.0
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    # Models
    process_model = ConstantVelocity(dim=2, process_noise=0.01)
    encoder_model = Encoder(state_dim=4, measure_position=True, dim=2, measurement_noise=0.05)
    gps_model = GPS(state_dim=4, dim=2, measurement_noise=2.0)

    # Filter
    ekf = EKF(process_model, {"encoder": encoder_model, "gps": gps_model}, initial_state)

    # Simulation: IMU at 400Hz, GPS at 5Hz
    dt_imu = 1.0 / 400.0  # 400Hz
    dt_gps = 1.0 / 5.0  # 5Hz

    duration = 5.0
    steps_imu = int(duration / dt_imu)

    true_velocity = np.array([2.0, 1.0])

    timestamps = []
    true_positions = []
    estimated_positions = []
    gps_updates = []

    timestamp = 0.0
    next_gps_time = 0.0

    for i in range(steps_imu):
        timestamp += dt_imu

        # Predict at IMU rate
        ekf.predict(u=np.array([]), dt=dt_imu, timestamp=timestamp)

        # True position
        true_pos = true_velocity * timestamp
        true_positions.append(true_pos.copy())
        timestamps.append(timestamp)

        # IMU/Encoder update (every step)
        encoder_z = true_pos + np.random.normal(0, 0.05, size=2)
        ekf.update(z=encoder_z, sensor_name="encoder", timestamp=timestamp)

        # GPS update (at lower rate)
        if timestamp >= next_gps_time:
            gps_z = true_pos + np.random.normal(0, 2.0, size=2)
            ekf.update(z=gps_z, sensor_name="gps", timestamp=timestamp)
            gps_updates.append((timestamp, gps_z.copy()))
            next_gps_time += dt_gps

        estimated_positions.append(ekf.state.x[:2].copy())

    # Plot results
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)
    gps_updates = np.array([gps[1] for gps in gps_updates])
    gps_times = [gps[0] for gps in gps_updates]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(true_positions[:, 0], true_positions[:, 1], "b-", label="True", linewidth=2)
    plt.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        "r--",
        label="Estimated",
        linewidth=1.5,
    )
    if len(gps_updates) > 0:
        plt.scatter(
            gps_updates[:, 0], gps_updates[:, 1], c="g", s=50, marker="x", label="GPS", zorder=5
        )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory (Async Sensors)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    error = np.linalg.norm(estimated_positions - true_positions, axis=1)
    plt.plot(timestamps, error, "g-", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title("Estimation Error")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(timestamps, estimated_positions[:, 0], "r-", label="Estimated X", linewidth=1.5)
    plt.plot(timestamps, true_positions[:, 0], "b--", label="True X", linewidth=2)
    if len(gps_times) > 0:
        gps_x = [gps[1][0] for gps in zip(gps_times, gps_updates)]
        plt.scatter(gps_times, gps_x, c="g", s=50, marker="x", label="GPS", zorder=5)
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (m)")
    plt.title("X Position Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("async_sensor_demo.png", dpi=150)
    print("Saved plot to async_sensor_demo.png")
    print(f"IMU rate: {1/dt_imu:.0f} Hz, GPS rate: {1/dt_gps:.0f} Hz")
    print(f"Final position error: {error[-1]:.3f} m")
    print(f"Final velocity estimate: {ekf.state.x[2:4]}")


if __name__ == "__main__":
    simulate_async_sensors()
