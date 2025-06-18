"""Feed-forward backbone network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _get_activation(name: str) -> type[nn.Module]:
    """Return activation class from name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'")
    return ACTIVATIONS[name]


@dataclass
class BackboneConfig:
    """Configuration for :class:`Backbone`."""

    in_dims: int
    hidden: Sequence[int] | None = (256, 128)
    activation: str = "gelu"
    dropout: float = 0.1


class Backbone(nn.Module):
    """Simple feed-forward backbone used by the heads."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = config.in_dims
        hidden = list(config.hidden) if config.hidden else []
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(_get_activation(config.activation)())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev_dim = h

        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
