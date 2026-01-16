# Estimator

[![PyPI version](https://badge.fury.io/py/estimator.svg)](https://badge.fury.io/py/estimator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-focused state estimation runtime for edge devices.**

This library provides a flexible and efficient framework for state estimation using various sensor fusion techniques, with a focus on running on resource-constrained edge devices.

## Features

*   **Extensible:** Easily add new process models, measurement models, and filters.
*   **Multiple Backends:** Supports both NumPy and JAX for numerical computations.
*   **Lightweight:** Designed to be minimal and efficient for edge deployment.
*   **Sensor Fusion:** Fuse measurements from multiple sensors with different data rates.

## Installation

Install the library from PyPI:

```bash
pip install estimator
```

To include the JAX backend, install the `jax` extra:

```bash
pip install estimator[jax]
```

## Usage

Here's a simple example of fusing synthetic GPS and encoder data to estimate the 2D position and velocity of a system with constant velocity.

```python
import numpy as np
from estimator.core.estimator import EKF
from estimator.core.state import State
from estimator.models.measurement.encoder import Encoder
from estimator.models.measurement.gps import GPS
from estimator.models.process.constant_velocity import ConstantVelocity

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

# Simulate
dt = 0.1
for i in range(100):
    t = i * dt
    ekf.predict(u=np.array([]), dt=dt, timestamp=t)

    # High-rate, low-noise encoder measurement
    true_pos = ekf.state.x[:2] + np.random.normal(0, 0.1, size=2)
    ekf.update(z=true_pos, sensor_name="encoder", timestamp=t)

    # Low-rate, high-noise GPS measurement
    if i % 10 == 0:
        true_pos_gps = ekf.state.x[:2] + np.random.normal(0, 1.0, size=2)
        ekf.update(z=true_pos_gps, sensor_name="gps", timestamp=t)

print(f"Final state: {ekf.state.x}")
```

## Running Tests

To run the test suite, first install the development dependencies:

```bash
pip install -e .[dev]
```

Then run pytest:

```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the [GitHub repository](https://github.com/Ankit-x1/estimator).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
