"""Data loading utilities."""

from .dummy import load_dummy
from .synthetic import generate_synthetic, get_synth_dataloaders

__all__ = ["load_dummy", "generate_synthetic", "get_synth_dataloaders"]
