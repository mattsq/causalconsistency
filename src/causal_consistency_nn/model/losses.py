"""Loss utilities for causal consistency models."""

from __future__ import annotations

import torch
from torch.distributions import Distribution, Normal, Categorical


def nll_gaussian(dist: Distribution, target: torch.Tensor) -> torch.Tensor:
    """Return the negative log likelihood under a Gaussian distribution."""
    if not isinstance(dist, Normal):
        raise TypeError("dist must be a Normal distribution")
    return -dist.log_prob(target).mean()


def cross_entropy(dist: Distribution, target: torch.Tensor) -> torch.Tensor:
    """Return the cross entropy between a categorical distribution and target labels."""
    if not isinstance(dist, Categorical):
        raise TypeError("dist must be a Categorical distribution")
    return -dist.log_prob(target).mean()


def entropy_categorical(dist: Distribution) -> torch.Tensor:
    """Return the entropy of a categorical distribution."""
    if not isinstance(dist, Categorical):
        raise TypeError("dist must be a Categorical distribution")
    return dist.entropy().mean()
