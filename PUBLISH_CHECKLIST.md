# Pre-Publishing Checklist

## Completed Fixes

- [x] **Version Sync**: Added dynamic version to `pyproject.toml` (reads from `__init__.py`)
- [x] **Type Hints**: Fixed mypy errors in `jax_backend.py` and `logging.py`
- [x] **GitHub Actions**: Added CI/CD workflow for testing, linting, and building
- [x] **CHANGELOG**: Created changelog file
- [x] **Tool Configuration**: Added black, ruff, and mypy configs to `pyproject.toml`
- [x] **Pre-commit**: Added pre-commit configuration

## Remaining Tasks

### Before Publishing

1. **Fix Code Style**
   ```bash
   python fix_code_style.py
   # Or manually:
   black estimator tests examples
   ruff check --fix estimator tests examples
   ```

2. **Verify README Encoding**
   ```bash
   # Check for null bytes
   python -c "with open('README.md', 'rb') as f: data = f.read(); print('Null bytes:', b'\\x00' in data)"
   # If null bytes found, recreate README
   ```

3. **Update Repository URLs**
   - Edit `pyproject.toml` lines 47-50
   - Replace `yourusername` with your actual GitHub username

4. **Run Full Test Suite**
   ```bash
   pytest -v
   python test_all.py
   ```

5. **Build and Verify Package**
   ```bash
   python -m build
   python -m twine check dist/*
   pip install dist/estimator-*.whl --force-reinstall
   ```

6. **Test on TestPyPI First**
   ```bash
   python -m twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ estimator
   ```

### Optional Improvements

- [ ] Add coverage badge to README
- [ ] Add PyPI version badge to README
- [ ] Add GitHub Actions status badge to README
- [ ] Create GitHub release notes
- [ ] Add API documentation (Sphinx/mkdocs)

## Quick Commands

```bash
# 1. Fix code style
python fix_code_style.py

# 2. Run tests
pytest -v

# 3. Build package
python -m build

# 4. Check package
python -m twine check dist/*

# 5. Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# 6. Upload to PyPI (after testing)
python -m twine upload dist/*
```

## Version Update Process

When releasing a new version:

1. Update `estimator/__init__.py`: `__version__ = "0.1.1"`
2. Update `CHANGELOG.md` with new version
3. Commit changes: `git commit -am "Bump version to 0.1.1"`
4. Tag release: `git tag v0.1.1`
5. Push: `git push && git push --tags`
6. Build and publish: `python -m build && python -m twine upload dist/*`

---

**Status**: Ready for code style fixes and final verification before publishing! 

