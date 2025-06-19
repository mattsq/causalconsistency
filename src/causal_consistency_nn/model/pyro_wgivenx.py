from __future__ import annotations

from pyro.nn import PyroModule
import torch
from torch import nn
from pyro.distributions import Normal

from .heads import WgivenXConfig


class PyroWgivenX(PyroModule):
    """Pyro module for ``p(W | X)``."""

    def __init__(self, cfg: WgivenXConfig) -> None:  # noqa: D401
        super().__init__()
        self.cfg = cfg
        self.fc = PyroModule[nn.Linear](cfg.h_dim, 2 * cfg.w_dim)

    def forward(self, h: torch.Tensor) -> Normal:
        out = self.fc(h)
        mu, log_sigma = out.chunk(2, dim=-1)
        return Normal(mu, log_sigma.exp())


__all__ = ["PyroWgivenX"]
