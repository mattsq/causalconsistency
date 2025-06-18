from __future__ import annotations

from pyro.nn import PyroModule
import torch
from torch import nn
from pyro.distributions import Categorical

from .heads import YgivenXZConfig


class PyroYgivenXZ(PyroModule):
    """Pyro module for ``p(Y | X,Z)``."""

    def __init__(self, cfg: YgivenXZConfig) -> None:  # noqa: D401
        super().__init__()
        self.cfg = cfg
        self.fc = PyroModule[nn.Linear](cfg.h_dim + cfg.z_dim, cfg.y_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> Categorical:
        logits = self.fc(torch.cat([h, z], dim=-1))
        return Categorical(logits=logits)


__all__ = ["PyroYgivenXZ"]
