"""
Example: Raspberry Pi IMU + Encoder demo.

Shows how to use the library on edge hardware.
Note: This is a simulation; replace with actual sensor drivers.
"""

import numpy as np

from estimator.core.estimator import EKF
from estimator.core.state import State
from estimator.gating.mahalanobis import MahalanobisGate
from estimator.models.measurement.encoder import Encoder
from estimator.models.process.constant_velocity import ConstantVelocity
from estimator.noise.adaptive import AdaptiveNoise


def simulate_rpi_sensors():
    """
    Simulate Raspberry Pi sensor setup.

    In real deployment, replace sensor reading with:
    - IMU: Read from I2C/SPI (e.g., MPU6050)
    - Encoder: Read GPIO interrupts
    """
    print("Raspberry Pi IMU + Encoder Demo")
    print("=" * 40)

    # Initial state: [px, py, vx, vy]
    initial_x = np.array([0.0, 0.0, 0.0, 0.0])
    initial_P = np.eye(4) * 1.0
    initial_state = State(initial_x, initial_P, timestamp=0.0)

    # Models
    process_model = ConstantVelocity(dim=2, process_noise=0.01)
    encoder_model = Encoder(state_dim=4, measure_position=True, dim=2, measurement_noise=0.05)

    # Adaptive noise for self-healing
    Q_init = process_model.noise(0.01)
    R_init = encoder_model.covariance()
    adaptive_noise = AdaptiveNoise(
        Q_init=Q_init, R_init=R_init, window_size=50, adaptation_rate=0.1
    )

    # Outlier rejection
    gate = MahalanobisGate(threshold=9.0, enabled=True)

    # Filter
    ekf = EKF(
        process_model,
        {"encoder": encoder_model},
        initial_state,
        adaptive_noise=adaptive_noise,
        gate=gate,
    )

    # Simulate sensor readings
    dt = 0.01  # 100Hz
    true_velocity = np.array([1.0, 0.5])

    print("\nSimulating sensor readings...")
    for i in range(100):
        t = i * dt

        # Predict
        ekf.predict(u=np.array([]), dt=dt, timestamp=t)

        # Simulate encoder reading (with occasional outliers)
        true_pos = true_velocity * t
        if i % 20 == 0:
            # Simulate outlier
            encoder_z = true_pos + np.random.normal(0, 10.0, size=2)  # Large noise
        else:
            encoder_z = true_pos + np.random.normal(0, 0.05, size=2)

        # Update (gate will reject outliers)
        ekf.update(z=encoder_z, sensor_name="encoder", timestamp=t)

        if i % 10 == 0:
            print(
                f"t={t:.2f}s: pos=({ekf.state.x[0]:.2f}, {ekf.state.x[1]:.2f}), "
                f"vel=({ekf.state.x[2]:.2f}, {ekf.state.x[3]:.2f}), "
                f"trace(P)={np.trace(ekf.state.P):.4f}"
            )

    # Print statistics
    print("\n" + "=" * 40)
    print("Statistics:")
    gate_stats = gate.get_stats()
    print(
        f"Gate: {gate_stats['accepted']} accepted, {gate_stats['rejected']} rejected "
        f"({gate_stats['rejection_rate']*100:.1f}% rejection rate)"
    )
    print(
        f"Final state: pos=({ekf.state.x[0]:.3f}, {ekf.state.x[1]:.3f}), "
        f"vel=({ekf.state.x[2]:.3f}, {ekf.state.x[3]:.3f})"
    )
    print(f"Final covariance trace: {np.trace(ekf.state.P):.6f}")

    print("\nNote: Replace sensor reading with actual hardware drivers:")
    print("  - IMU: Read from I2C/SPI (e.g., MPU6050)")
    print("  - Encoder: Read GPIO interrupts")


if __name__ == "__main__":
    simulate_rpi_sensors()
