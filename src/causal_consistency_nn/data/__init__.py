"""Data loading utilities."""

from .dummy import load_dummy
from .synthetic import SynthConfig, make_synthetic_loaders

__all__ = ["load_dummy", "SynthConfig", "make_synthetic_loaders"]
