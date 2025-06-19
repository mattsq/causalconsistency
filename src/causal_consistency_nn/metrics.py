from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor
from torch.distributions import Normal
import torch.nn.functional as F


def average_treatment_effect(
    y: Tensor, z: Tensor, treat: int = 1, control: int = 0
) -> Tensor:
    """Return the Average Treatment Effect from outcome samples."""
    t_mask = y == treat
    c_mask = y == control
    if t_mask.sum() == 0 or c_mask.sum() == 0:
        raise ValueError("both treatment groups must have at least one sample")
    return z[t_mask].mean() - z[c_mask].mean()


def log_likelihood_normal(mu: Tensor, sigma: Tensor, target: Tensor) -> Tensor:
    """Compute mean log likelihood of a Normal distribution."""
    dist = Normal(mu, sigma)
    return dist.log_prob(target).mean()


def dataset_log_likelihood(
    model: torch.nn.Module,
    loader: Iterable[tuple[Tensor, ...]],
) -> float:
    """Average log likelihood of ``Z`` given ``X`` and ``Y`` for a dataset."""
    total = 0.0
    count = 0
    for batch in loader:
        if len(batch) == 4:
            _, x, y, z = batch
        else:
            x, y, z = batch
        h = model.backbone(x)
        y_oh = F.one_hot(y, num_classes=model.y_dim).float()
        dist = model.head_z(h, y_oh)
        total += dist.log_prob(z).sum().item()
        count += z.numel()
    return total / count


__all__ = [
    "average_treatment_effect",
    "log_likelihood_normal",
    "dataset_log_likelihood",
]
