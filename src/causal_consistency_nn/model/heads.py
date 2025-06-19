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


@dataclass
class WgivenXConfig:
    """Configuration for :class:`WgivenX`."""

    h_dim: int
    w_dim: int


class ZgivenXY(nn.Module):
    """Distribution of ``Z`` given ``X`` and ``Y``."""

    def __init__(self, cfg: ZgivenXYConfig) -> None:  # noqa: D401
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.h_dim + cfg.y_dim, 2 * cfg.z_dim)

    def forward(self, h: torch.Tensor, y_onehot: torch.Tensor) -> Normal:
        """Return ``p(Z | X,Y)`` as a Normal distribution."""
        if h.shape[-1] != self.cfg.h_dim:
            raise ValueError(
                f"h last dimension {h.shape[-1]} does not match h_dim {self.cfg.h_dim}"
            )
        if y_onehot.shape[-1] != self.cfg.y_dim:
            raise ValueError(
                f"y_onehot last dimension {y_onehot.shape[-1]} "
                f"does not match y_dim {self.cfg.y_dim}"
            )
        if h.shape[:-1] != y_onehot.shape[:-1]:
            raise ValueError("Batch dimensions of h and y_onehot must match")

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
        if h.shape[-1] != self.cfg.h_dim:
            raise ValueError(
                f"h last dimension {h.shape[-1]} does not match h_dim {self.cfg.h_dim}"
            )
        if z.shape[-1] != self.cfg.z_dim:
            raise ValueError(
                f"z last dimension {z.shape[-1]} does not match z_dim {self.cfg.z_dim}"
            )
        if h.shape[:-1] != z.shape[:-1]:
            raise ValueError("Batch dimensions of h and z must match")

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
        if h.shape[-1] != self.cfg.h_dim:
            raise ValueError(
                f"h last dimension {h.shape[-1]} does not match h_dim {self.cfg.h_dim}"
            )
        if y_onehot.shape[-1] != self.cfg.y_dim:
            raise ValueError(
                f"y_onehot last dimension {y_onehot.shape[-1]} "
                f"does not match y_dim {self.cfg.y_dim}"
            )
        if h.shape[:-1] != y_onehot.shape[:-1]:
            raise ValueError("Batch dimensions of h and y_onehot must match")

        out = self.fc(torch.cat([h, y_onehot], dim=-1))
        mu, log_sigma = out.chunk(2, dim=-1)
        return Normal(mu, log_sigma.exp())


class WgivenX(nn.Module):
    """Distribution of ``W`` given ``X``."""

    def __init__(self, cfg: WgivenXConfig) -> None:  # noqa: D401
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.h_dim, 2 * cfg.w_dim)

    def forward(self, h: torch.Tensor) -> Normal:
        if h.shape[-1] != self.cfg.h_dim:
            raise ValueError(
                f"h last dimension {h.shape[-1]} does not match h_dim {self.cfg.h_dim}"
            )
        out = self.fc(h)
        mu, log_sigma = out.chunk(2, dim=-1)
        return Normal(mu, log_sigma.exp())


__all__ = [
    "ZgivenXY",
    "YgivenXZ",
    "XgivenYZ",
    "WgivenX",
    "ZgivenXYConfig",
    "YgivenXZConfig",
    "XgivenYZConfig",
    "WgivenXConfig",
]
