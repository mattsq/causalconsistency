"""Inference helpers for trained models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .train import ConsistencyModel


def predict_z(
    model: ConsistencyModel, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Return ``E[Z | X=x, Y=y]``."""
    model.eval()
    with torch.no_grad():
        return model.head_z_given_xy(x, y)


def counterfactual_z(
    model: ConsistencyModel, x: torch.Tensor, y_prime: torch.Tensor
) -> torch.Tensor:
    """Estimate ``Z`` for counterfactual treatment ``y_prime``."""
    return predict_z(model, x, y_prime)


def impute_y(model: ConsistencyModel, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return posterior probabilities of ``Y`` given ``X`` and ``Z``."""
    model.eval()
    with torch.no_grad():
        logits = model.head_y_given_xz(x, z)
        return F.softmax(logits, dim=-1)


__all__ = ["predict_z", "counterfactual_z", "impute_y"]
