from __future__ import annotations

from pyro.nn import PyroModule
import torch
from torch import nn
from pyro.distributions import Normal

from .heads import XgivenYZConfig


class PyroXgivenYZ(PyroModule):
    """Pyro module for ``p(X | Y,Z)``."""

    def __init__(self, cfg: XgivenYZConfig) -> None:  # noqa: D401
        super().__init__()
        self.cfg = cfg
        self.fc = PyroModule[nn.Linear](cfg.h_dim + cfg.y_dim, 2 * cfg.x_dim)

    def forward(self, h: torch.Tensor, y_onehot: torch.Tensor) -> Normal:
        out = self.fc(torch.cat([h, y_onehot], dim=-1))
        mu, log_sigma = out.chunk(2, dim=-1)
        return Normal(mu, log_sigma.exp())


__all__ = ["PyroXgivenYZ"]
