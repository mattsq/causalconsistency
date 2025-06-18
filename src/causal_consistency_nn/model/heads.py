"""Neural network heads producing conditional distributions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical, Normal


@dataclass
class ZgivenXYConfig:
    """Configuration for :class:`ZgivenXY`."""

    h_dim: int
    y_dim: int
    z_dim: int


@dataclass
class YgivenXZConfig:
    """Configuration for :class:`YgivenXZ`."""

    h_dim: int
    z_dim: int
    y_dim: int


@dataclass
class XgivenYZConfig:
    """Configuration for :class:`XgivenYZ`."""

    h_dim: int
    y_dim: int
    x_dim: int


class ZgivenXY(nn.Module):
    """Distribution of ``Z`` given ``X`` and ``Y``."""

    def __init__(self, cfg: ZgivenXYConfig) -> None:  # noqa: D401
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.h_dim + cfg.y_dim, 2 * cfg.z_dim)

    def forward(self, h: torch.Tensor, y_onehot: torch.Tensor) -> Normal:
        """Return ``p(Z | X,Y)`` as a Normal distribution."""
        out = self.fc(torch.cat([h, y_onehot], dim=-1))
        mu, log_sigma = out.chunk(2, dim=-1)
        return Normal(mu, log_sigma.exp())


class YgivenXZ(nn.Module):
    """Distribution of ``Y`` given ``X`` and ``Z``."""

    def __init__(self, cfg: YgivenXZConfig) -> None:  # noqa: D401
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.h_dim + cfg.z_dim, cfg.y_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> Categorical:
        """Return ``p(Y | X,Z)`` as a Categorical distribution."""
        logits = self.fc(torch.cat([h, z], dim=-1))
        return Categorical(logits=logits)


class XgivenYZ(nn.Module):
    """Distribution of ``X`` given ``Y`` and ``Z``."""

    def __init__(self, cfg: XgivenYZConfig) -> None:  # noqa: D401
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.h_dim + cfg.y_dim, 2 * cfg.x_dim)

    def forward(self, h: torch.Tensor, y_onehot: torch.Tensor) -> Normal:
        """Return ``p(X | Y,Z)`` as a Normal distribution."""
        out = self.fc(torch.cat([h, y_onehot], dim=-1))
        mu, log_sigma = out.chunk(2, dim=-1)
        return Normal(mu, log_sigma.exp())


__all__ = [
    "ZgivenXY",
    "YgivenXZ",
    "XgivenYZ",
    "ZgivenXYConfig",
    "YgivenXZConfig",
    "XgivenYZConfig",
]
