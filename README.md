# EstEdge

Production-focused state estimation runtime for edge devices.

## Overview

EstEdge is a lightweight, efficient library for state estimation and sensor fusion designed specifically for resource-constrained edge devices. It provides implementations of Kalman filters (KF, EKF, UKF) with support for multiple backends and extensible sensor models.

## Features

- **Multiple Filter Types**: Kalman Filter (KF), Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF)
- **Backend Support**: NumPy for production, JAX for research and auto-differentiation
- **Sensor Fusion**: Multi-sensor integration with different update rates
- **Extensible Models**: Modular process and measurement model system
- **Edge Optimized**: Minimal dependencies and efficient computations
- **Type Safety**: Full type annotations and MyPy compatibility

## Installation

### Basic Installation

```bash
pip install estedge
```

### With JAX Backend

```bash
pip install estedge[jax]
```

### Development Installation

```bash
git clone https://github.com/Ankit-x1/estimator.git
cd estimator
pip install -e .[dev]
```

## Quick Start

### Basic Example

```python
import numpy as np
from estedge import EKF, State
from estedge.models.measurement import Encoder
from estedge.models.process import ConstantVelocity

# Initialize state [position, velocity]
initial_x = np.array([0.0, 1.0])
initial_P = np.eye(2) * 0.1
state = State(initial_x, initial_P, timestamp=0.0)

# Configure models
process_model = ConstantVelocity(dim=1)
measurement_model = Encoder(state_dim=2, measure_position=True, dim=1)

# Create filter
ekf = EKF(process_model, {"encoder": measurement_model}, state)

# Prediction step
ekf.predict(u=np.array([]), dt=0.1, timestamp=0.1)

# Update step
measurement = np.array([0.15])  # Position measurement
ekf.update(z=measurement, sensor_name="encoder", timestamp=0.1)

print(f"Estimated state: {ekf.state.x}")
```

### Multi-Sensor Fusion

```python
from estedge.models.measurement import GPS, IMU

# Multiple measurement models
measurement_models = {
    "gps": GPS(state_dim=4, dim=2),
    "imu": IMU(state_dim=6, dim=3),
    "encoder": Encoder(state_dim=4, measure_position=True, dim=2)
}

ekf = EKF(process_model, measurement_models, initial_state)

# Update with different sensors at different rates
ekf.update(z=gps_measurement, sensor_name="gps", timestamp=t)
ekf.update(z=imu_measurement, sensor_name="imu", timestamp=t)
ekf.update(z=encoder_measurement, sensor_name="encoder", timestamp=t)
```

## Architecture

### Core Components

- **Filters**: KF, EKF, UKF implementations
- **State**: State vector and covariance management
- **Models**: Process and measurement model abstractions
- **Backends**: NumPy and JAX computational backends
- **Noise**: Static and adaptive noise models
- **Utils**: Linear algebra utilities and validation functions

### Available Models

#### Process Models
- `ConstantVelocity`: Linear constant velocity dynamics
- `ConstantAcceleration`: Linear constant acceleration dynamics
- `IMUKinematics`: IMU-based kinematic modeling

#### Measurement Models
- `GPS`: GPS position measurements
- `IMU`: IMU acceleration and angular velocity
- `Encoder`: Encoder position/velocity measurements
- `Magnetometer`: Magnetic field measurements

## Backend Selection

### NumPy Backend (Default)

```python
from estedge.backend import get_backend

backend = get_backend("numpy")
```

### JAX Backend

```python
backend = get_backend("jax")
# Enables automatic differentiation and JIT compilation
```

## Testing

Run the complete test suite:

```bash
pytest
```

Run specific test categories:

```bash
pytest tests/test_convergence.py
pytest tests/test_stability.py
pytest tests/test_async_sensors.py
```

## Development

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting and code analysis
- **MyPy**: Static type checking

Run all quality checks:

```bash
black --check estedge tests examples
ruff check estedge tests examples
mypy estedge
```


## Performance

EstEdge is optimized for edge devices:

- **Memory Efficient**: Minimal memory allocation during filtering
- **Computationally Light**: Optimized linear algebra operations
- **Scalable**: Handles high-rate sensor data efficiently
- **Portable**: Pure Python implementation with minimal dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Guidelines

- Follow the existing code style (Black formatting)
- Add type annotations to new functions
- Include docstrings for public APIs
- Add unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use EstEdge in your research, please cite:

```bibtex
@software{estedge,
  title={EstEdge: Production-focused state estimation runtime for edge devices},
  author={EstEdge Contributors},
  year={2024},
  url={https://github.com/Ankit-x1/estimator}
}
```