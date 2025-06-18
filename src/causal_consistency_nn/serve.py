"""Lightweight inference helpers."""

from __future__ import annotations

from typing import Any

from torch import Tensor


def predict_z(model: Any, x: Tensor, y: Tensor) -> Tensor:
    """Return the mean of ``Z | X=x, Y=y``."""
    dist = model.head_z_given_xy(x, y)
    return dist.mean


def counterfactual_z(model: Any, x: Tensor, y_prime: Tensor) -> Tensor:
    """Estimate counterfactual ``Z`` for a different treatment ``y_prime``."""
    dist = model.head_z_given_xy(x, y_prime)
    return dist.mean


def impute_y(model: Any, x: Tensor, z: Tensor) -> Tensor:
    """Return posterior probabilities ``p(Y | X=x, Z=z)``."""
    dist = model.head_y_given_xz(x, z)
    return dist.probs
