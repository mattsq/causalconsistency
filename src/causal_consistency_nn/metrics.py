"""Causal evaluation metrics."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.distributions import Distribution


def average_treatment_effect(
    model: Any, x: Tensor, treatment0: int | Tensor = 0, treatment1: int | Tensor = 1
) -> Tensor:
    """Estimate the Average Treatment Effect via the ``Z|X,Y`` head."""
    if isinstance(treatment0, int):
        y0 = torch.full((x.size(0),), treatment0, dtype=torch.long, device=x.device)
    else:
        y0 = treatment0
    if isinstance(treatment1, int):
        y1 = torch.full((x.size(0),), treatment1, dtype=torch.long, device=x.device)
    else:
        y1 = treatment1

    dist0 = model.head_z_given_xy(x, y0)
    dist1 = model.head_z_given_xy(x, y1)
    return (dist1.mean - dist0.mean).mean()


def log_likelihood(dist: Distribution, target: Tensor) -> Tensor:
    """Return the mean log likelihood under ``dist``."""
    return dist.log_prob(target).mean()
