# Testing and Publishing Guide for edge-estimators

## 📋 Pre-Testing Checklist

Before running tests, ensure you have:

1. ✅ **Python 3.8+ installed**
2. ✅ **Virtual environment created** (recommended)
3. ✅ **Dependencies installed**
4. ✅ **Code is clean** (no syntax errors)

---

## 🧪 Step 1: Setup Testing Environment

### 1.1 Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 1.2 Install Dependencies

```bash
# Install the library in development mode
pip install -e .

# Install development dependencies (pytest, etc.)
pip install -e ".[dev]"

# Optional: Install JAX backend for testing
pip install -e ".[jax]"
```

### 1.3 Verify Installation

```bash
python -c "import edge_estimators; print(edge_estimators.__version__)"
```

---

## 🧪 Step 2: Run All Tests

### 2.1 Run All Tests with pytest

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=edge_estimators --cov-report=html

# Run specific test file
pytest tests/test_convergence.py

# Run specific test function
pytest tests/test_convergence.py::test_kf_convergence
```

### 2.2 Test Categories

Your test suite includes:

- **`test_convergence.py`**: Tests filter convergence under noise
- **`test_stability.py`**: Tests numerical stability (PSD, NaN handling)
- **`test_dropout.py`**: Tests missing measurements
- **`test_async_sensors.py`**: Tests asynchronous sensor fusion

### 2.3 Expected Test Results

All tests should pass. If any fail:
1. Check error messages
2. Verify dependencies are installed
3. Check Python version (3.8+)

---

## 🔍 Step 3: Manual Testing

### 3.1 Test README Examples

Run the examples from README to ensure they work:

```bash
# Test basic example
python -c "
import numpy as np
from edge_estimators import EKF, State
from edge_estimators.models.process.constant_velocity import ConstantVelocity
from edge_estimators.models.measurement.encoder import Encoder

initial_x = np.array([0.0, 1.0])
initial_P = np.eye(2) * 0.1
initial_state = State(initial_x, initial_P, timestamp=0.0)

process_model = ConstantVelocity(dim=1, process_noise=0.01)
encoder_model = Encoder(state_dim=2, measure_position=True, dim=1, measurement_noise=0.1)

ekf = EKF(process_model, {'encoder': encoder_model}, initial_state)

dt = 0.1
for i in range(10):
    ekf.predict(u=np.array([]), dt=dt)
    measurement = np.array([i * dt + np.random.normal(0, 0.1)])
    ekf.update(z=measurement, sensor_name='encoder')
    print(f'Step {i}: Position={ekf.state.x[0]:.3f}, Velocity={ekf.state.x[1]:.3f}')

print('✅ Basic example works!')
"
```

### 3.2 Test Backend Switching

```bash
python -c "
from edge_estimators.backend import get_backend

# Test NumPy backend
backend_np = get_backend('numpy')
print('✅ NumPy backend works')

# Test JAX backend (if installed)
try:
    backend_jax = get_backend('jax')
    print('✅ JAX backend works')
except ImportError:
    print('⚠️  JAX not installed (optional)')
"
```

### 3.3 Test All Models

```bash
python -c "
from edge_estimators.models.process import ConstantVelocity, ConstantAcceleration, IMUKinematics
from edge_estimators.models.measurement import Encoder, IMU, GPS, Magnetometer

# Test process models
cv = ConstantVelocity(dim=1)
ca = ConstantAcceleration(dim=1)
imu_k = IMUKinematics(dim=3)
print('✅ All process models import correctly')

# Test measurement models
enc = Encoder(state_dim=2, measure_position=True, dim=1)
imu_m = IMU(state_dim=6, dim=3)
gps = GPS(state_dim=4, dim=2)
mag = Magnetometer(state_dim=5)
print('✅ All measurement models import correctly')
"
```

### 3.4 Test Examples

```bash
# Run example scripts
python examples/async_sensor_demo.py
python examples/imu_encoder_sim.py
# Note: rpi_imu_encoder.py may require hardware
```

---

## 🧹 Step 4: Code Quality Checks

### 4.1 Linting

```bash
# Install flake8 if not already installed
pip install flake8

# Run flake8
flake8 edge_estimators --max-line-length=100 --ignore=E203,W503

# Or use black for formatting
pip install black
black --check edge_estimators
```

### 4.2 Type Checking (Optional)

```bash
# Install mypy
pip install mypy

# Run type checking
mypy edge_estimators --ignore-missing-imports
```

---

## 📦 Step 5: Build the Package

### 5.1 Install Build Tools

```bash
pip install build twine
```

### 5.2 Build Distribution

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source and wheel distributions
python -m build

# This creates:
# - dist/edge_estimators-0.1.0.tar.gz (source distribution)
# - dist/edge_estimators-0.1.0-py3-none-any.whl (wheel)
```

### 5.3 Verify Build

```bash
# Check the built package
python -m twine check dist/*

# Test install from local build
pip install dist/edge_estimators-0.1.0-py3-none-any.whl --force-reinstall

# Test import
python -c "import edge_estimators; print(edge_estimators.__version__)"
```

---

## 🚀 Step 6: Publish to PyPI

### 6.1 Update Version

Before publishing, update version in:
- `pyproject.toml` (line 7)
- `edge_estimators/__init__.py` (line 10)

```python
# In pyproject.toml
version = "0.1.0"  # Update to next version

# In edge_estimators/__init__.py
__version__ = "0.1.0"  # Update to match
```

### 6.2 Update Repository URLs

Update URLs in `pyproject.toml` (lines 47-50) with your actual GitHub repository:

```toml
[project.urls]
Homepage = "https://github.com/Ankit-x1/estimator"
Documentation = "https://github.com/YOUR_USERNAME/edge-estimators"
Repository = "https://github.com/Ankit-x1/estimator"
Issues = "https://github.com/YOUR_USERNAME/edge-estimators/issues"
```

### 6.3 Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Create account
3. Verify email

### 6.4 Upload to TestPyPI First (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ edge-estimators
```

### 6.5 Upload to PyPI

```bash
# Upload to PyPI (production)
python -m twine upload dist/*

# You'll be prompted for:
# - Username: your_pypi_username
# - Password: your_pypi_password (or API token)
```

### 6.6 Verify Publication

```bash
# Wait a few minutes, then test install
pip install edge-estimators

# Test import
python -c "import edge_estimators; print(edge_estimators.__version__)"
```

---

## 📝 Step 7: Post-Publication

### 7.1 Create GitHub Release

1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Tag: `v0.1.0`
4. Title: `edge-estimators v0.1.0`
5. Description: Release notes
6. Publish release

### 7.2 Update Documentation

- Update README if needed
- Add changelog
- Update examples if API changed

---

## 🐛 Troubleshooting

### Tests Fail

```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install --upgrade -e ".[dev]"

# Run with more verbose output
pytest -vv
```

### Build Fails

```bash
# Clean and rebuild
rm -rf dist/ build/ *.egg-info
python -m build
```

### Upload Fails

```bash
# Check credentials
python -m twine check dist/*

# Use API token instead of password
# Generate at: https://pypi.org/manage/account/token/
```

### Import Errors After Install

```bash
# Uninstall and reinstall
pip uninstall edge-estimators -y
pip install edge-estimators

# Check installation
pip show edge-estimators
```

---

## ✅ Final Checklist Before Publishing

- [ ] All tests pass (`pytest`)
- [ ] Code quality checks pass (`flake8`, `black`)
- [ ] README examples work
- [ ] Version updated in both files
- [ ] Repository URLs updated
- [ ] Package builds successfully (`python -m build`)
- [ ] Package installs and imports correctly
- [ ] Tested on TestPyPI first
- [ ] Ready to publish to PyPI

---

## 📚 Additional Resources

- [PyPI Packaging Guide](https://packaging.python.org/en/latest/)
- [pytest Documentation](https://docs.pytest.org/)
- [Python Packaging User Guide](https://packaging.python.org/guides/)

---

## 🎯 Quick Commands Summary

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -e ".[dev]"

# Test
pytest -v
pytest --cov=edge_estimators

# Build
python -m build
python -m twine check dist/*

# Publish (TestPyPI first)
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*  # Production
```

Good luck with your publication! 🚀

