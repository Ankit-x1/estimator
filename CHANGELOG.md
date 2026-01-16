# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-15

### Added
- Initial release of estimator
- **Core Estimators**: KF (Linear Kalman Filter), EKF (Extended Kalman Filter), UKF (Unscented Kalman Filter)
- **Process Models**: ConstantVelocity, ConstantAcceleration, IMUKinematics
- **Measurement Models**: Encoder, IMU, GPS, Magnetometer
- **Backend Abstraction**: NumPy backend (production), JAX backend (optional, for research)
- **Adaptive Noise**: Innovation-based adaptive Q/R estimation for self-healing filters
- **Mahalanobis Gating**: Outlier rejection for sensor measurements
- **Numerical Stability**: Joseph form covariance updates, PSD enforcement, NaN/Inf handling
- **Asynchronous Sensor Fusion**: Support for variable-frequency sensors
- Comprehensive test suite covering convergence, stability, dropout, and async sensors
- Example scripts demonstrating usage patterns

### Features
- Sensor-first API design
- Production-ready for edge devices (Raspberry Pi, Jetson, MCU logs)
- Numerically robust with automatic PSD enforcement
- Extensible model system via base classes
- Multi-sensor support with per-sensor adaptive noise

### Documentation
- Comprehensive README with examples
- API documentation in docstrings
- Testing and publishing guide

---

## [Unreleased]

### Planned
- Additional process models
- More measurement models
- Performance optimizations
- Extended documentation

