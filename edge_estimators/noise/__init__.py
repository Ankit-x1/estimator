"""Noise models: static and adaptive."""

from edge_estimators.noise.adaptive import AdaptiveNoise
from edge_estimators.noise.static import StaticNoise

__all__ = ["StaticNoise", "AdaptiveNoise"]
