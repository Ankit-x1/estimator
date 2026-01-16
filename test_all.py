#!/usr/bin/env python
"""
Quick test script to verify all components work before publishing.
Run this before publishing: python test_all.py
"""

import sys

import numpy as np


def test_imports():
    """Test all main imports."""
    print("Testing imports...")
    try:
        from estimator import EKF, KF, UKF, State
        from estimator.backend import get_backend
        from estimator.gating import MahalanobisGate
        from estimator.models.measurement import (
            GPS,
            IMU,
            Encoder,
            Magnetometer,
        )
        from estimator.models.process import (
            ConstantAcceleration,
            ConstantVelocity,
            IMUKinematics,
        )
        from estimator.noise import AdaptiveNoise, StaticNoise

        # Instantiate each imported class to mark it as "used"
        _ = KF
        _ = EKF
        _ = UKF
        _ = State
        _ = get_backend
        _ = MahalanobisGate
        _ = Encoder
        _ = GPS
        _ = IMU
        _ = Magnetometer
        _ = ConstantAcceleration
        _ = ConstantVelocity
        _ = IMUKinematics
        _ = AdaptiveNoise
        _ = StaticNoise

        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_basic_kf():
    """Test basic KF functionality."""
    print("\nTesting KF...")
    try:
        from estimator import KF, State
        from estimator.models.measurement.encoder import Encoder
        from estimator.models.process.constant_velocity import ConstantVelocity

        initial_x = np.array([0.0, 1.0])
        initial_P = np.eye(2) * 0.1
        initial_state = State(initial_x, initial_P, timestamp=0.0)

        process_model = ConstantVelocity(dim=1, process_noise=0.01)
        measurement_model = Encoder(
            state_dim=2, measure_position=True, dim=1, measurement_noise=0.1
        )

        kf = KF(process_model, measurement_model, initial_state)

        # Run a few steps
        for i in range(5):
            kf.predict(u=np.array([]), dt=0.1)
            kf.update(z=np.array([i * 0.1 + np.random.normal(0, 0.1)]))

        assert not np.any(np.isnan(kf.state.x)), "State contains NaN"
        assert not np.any(np.isnan(kf.state.P)), "Covariance contains NaN"
        print("[OK] KF works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] KF test failed: {e}")
        return False


def test_basic_ekf():
    """Test basic EKF functionality."""
    print("\nTesting EKF...")
    try:
        from estimator import EKF, State
        from estimator.models.measurement.encoder import Encoder
        from estimator.models.process.constant_velocity import ConstantVelocity

        initial_x = np.array([0.0, 1.0])
        initial_P = np.eye(2) * 0.1
        initial_state = State(initial_x, initial_P, timestamp=0.0)

        process_model = ConstantVelocity(dim=1, process_noise=0.01)
        encoder_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

        ekf = EKF(process_model, {"encoder": encoder_model}, initial_state)

        # Run a few steps
        for i in range(5):
            ekf.predict(u=np.array([]), dt=0.1)
            ekf.update(z=np.array([i * 0.1 + np.random.normal(0, 0.1)]), sensor_name="encoder")

        assert not np.any(np.isnan(ekf.state.x)), "State contains NaN"
        assert not np.any(np.isnan(ekf.state.P)), "Covariance contains NaN"
        print("[OK] EKF works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] EKF test failed: {e}")
        return False


def test_basic_ukf():
    """Test basic UKF functionality."""
    print("\nTesting UKF...")
    try:
        from estimator import UKF, State
        from estimator.models.measurement.encoder import Encoder
        from estimator.models.process.constant_velocity import ConstantVelocity

        initial_x = np.array([0.0, 1.0])
        initial_P = np.eye(2) * 0.1
        initial_state = State(initial_x, initial_P, timestamp=0.0)

        process_model = ConstantVelocity(dim=1, process_noise=0.01)
        encoder_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

        ukf = UKF(process_model, {"encoder": encoder_model}, initial_state)

        # Run a few steps
        for i in range(5):
            ukf.predict(u=np.array([]), dt=0.1)
            ukf.update(z=np.array([i * 0.1 + np.random.normal(0, 0.1)]), sensor_name="encoder")

        assert not np.any(np.isnan(ukf.state.x)), "State contains NaN"
        assert not np.any(np.isnan(ukf.state.P)), "Covariance contains NaN"
        print("[OK] UKF works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] UKF test failed: {e}")
        return False


def test_backend():
    """Test backend functionality."""
    print("\nTesting backend...")
    try:
        from estimator.backend import get_backend

        # Test NumPy backend
        backend = get_backend("numpy")
        A = np.eye(3)
        B = backend.inv(A)
        assert np.allclose(A, B), "Backend inverse failed"
        print("[OK] NumPy backend works")

        # Test JAX backend (if available)
        try:
            get_backend("jax")
            print("[OK] JAX backend available")
        except ImportError:
            print("[WARN] JAX backend not installed (optional)")

        return True
    except Exception as e:
        print(f"[FAIL] Backend test failed: {e}")
        return False


def test_multi_sensor():
    """Test multi-sensor EKF."""
    print("\nTesting multi-sensor EKF...")
    try:
        from estimator import EKF, State
        from estimator.models.measurement.encoder import Encoder
        from estimator.models.measurement.gps import GPS
        from estimator.models.process.constant_velocity import ConstantVelocity

        initial_x = np.array([0.0, 0.0, 2.0, 1.0])  # [px, py, vx, vy]
        initial_P = np.eye(4) * 0.1
        initial_state = State(initial_x, initial_P, timestamp=0.0)

        process_model = ConstantVelocity(dim=2, process_noise=0.01)
        encoder_model = Encoder(state_dim=4, measure_position=True, dim=2, measurement_noise=0.05)
        gps_model = GPS(state_dim=4, dim=2, measurement_noise=2.0)

        ekf = EKF(process_model, {"encoder": encoder_model, "gps": gps_model}, initial_state)

        # Run a few steps
        for i in range(5):
            ekf.predict(u=np.array([]), dt=0.1)
            ekf.update(
                z=np.array([i * 0.2, i * 0.1]) + np.random.normal(0, 0.05, 2), sensor_name="encoder"
            )
            if i % 2 == 0:  # GPS every other step
                ekf.update(
                    z=np.array([i * 0.2, i * 0.1]) + np.random.normal(0, 2.0, 2), sensor_name="gps"
                )

        assert not np.any(np.isnan(ekf.state.x)), "State contains NaN"
        print("[OK] Multi-sensor EKF works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Multi-sensor test failed: {e}")
        return False


def test_adaptive_noise():
    """Test adaptive noise."""
    print("\nTesting adaptive noise...")
    try:
        from estimator import EKF, State
        from estimator.models.measurement.encoder import Encoder
        from estimator.models.process.constant_velocity import ConstantVelocity
        from estimator.noise.adaptive import AdaptiveNoise

        initial_x = np.array([0.0, 1.0])
        initial_P = np.eye(2) * 0.1
        initial_state = State(initial_x, initial_P, timestamp=0.0)

        process_model = ConstantVelocity(dim=1, process_noise=0.01)
        encoder_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

        Q_init = process_model.noise(0.01)
        R_init = encoder_model.covariance()
        adaptive_noise = AdaptiveNoise(Q_init=Q_init, R_init=R_init, window_size=10)

        ekf = EKF(
            process_model, {"encoder": encoder_model}, initial_state, adaptive_noise=adaptive_noise
        )

        # Run a few steps
        for i in range(10):
            ekf.predict(u=np.array([]), dt=0.1)
            ekf.update(z=np.array([i * 0.1 + np.random.normal(0, 0.1)]), sensor_name="encoder")

        assert not np.any(np.isnan(ekf.state.x)), "State contains NaN"
        print("[OK] Adaptive noise works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Adaptive noise test failed: {e}")
        return False


def test_all_models():
    """Test all model classes can be instantiated."""
    print("\nTesting all models...")
    try:
        from estimator.models.measurement import GPS, IMU, Encoder, Magnetometer
        from estimator.models.process import (
            ConstantAcceleration,
            ConstantVelocity,
            IMUKinematics,
        )

        # Process models
        ConstantVelocity(dim=1)
        ConstantAcceleration(dim=1)
        IMUKinematics(dim=3)

        # Measurement models
        Encoder(state_dim=2, measure_position=True, dim=1)
        # IMU needs state_dim=9 for [p, v, a] with dim=3 (3*3=9)
        IMU(state_dim=9, dim=3, acceleration_index=6)  # For ConstantAcceleration [p, v, a]
        GPS(state_dim=4, dim=2)
        Magnetometer(state_dim=5)  # [x, y, vx, vy, heading]

        print("[OK] All models instantiate correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Model instantiation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("edge-estimators Library Test Suite")
    print("=" * 60)

    tests = [
        test_imports,
        test_basic_kf,
        test_basic_ekf,
        test_basic_ukf,
        test_backend,
        test_multi_sensor,
        test_adaptive_noise,
        test_all_models,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("[OK] All tests passed! Library is ready for publishing.")
        return 0
    else:
        print("[FAIL] Some tests failed. Fix issues before publishing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
