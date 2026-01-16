#!/usr/bin/env python
"""
Script to fix code style issues before publishing.
Run: python fix_code_style.py
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] {description} passed")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"[FAIL] {description} failed")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] {description} crashed: {e}")
        return False


def main():
    """Run all code style fixes."""
    print("=" * 60)
    print("Code Style Fix Script")
    print("=" * 60)

    # Check if tools are installed
    tools = {
        "black": "pip install black",
        "ruff": "pip install ruff",
    }

    for tool, install_cmd in tools.items():
        result = subprocess.run(f"{tool} --version", shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"\n[WARN] {tool} not found. Install with: {install_cmd}")
            response = input(f"Install {tool} now? (y/n): ")
            if response.lower() == "y":
                subprocess.run(install_cmd.split(), check=False)

    # Run fixes
    fixes = [
        ("black estimator tests examples", "Format code with Black"),
        ("ruff check --fix estimator tests examples", "Fix linting issues with Ruff"),
        ("black --check estimator tests examples", "Verify Black formatting"),
        ("ruff check estimator tests examples", "Verify Ruff checks"),
    ]

    results = []
    for cmd, desc in fixes:
        results.append(run_command(cmd, desc))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[OK] All code style checks passed!")
        return 0
    else:
        print("\n[FAIL] Some checks failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
