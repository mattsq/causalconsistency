"""Synthetic dataset utilities for integration tests and examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SynthConfig:
    """Configuration for :func:`make_synthetic_loaders`."""

    n: int = 100
    batch_size: int = 32


def make_synthetic_loaders(
    cfg: SynthConfig | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return supervised and unsupervised loaders for a toy SCM."""
    if cfg is None:
        cfg = SynthConfig()

    x = torch.randn(cfg.n, 1)
    y = (x.squeeze() > 0).long()
    z = x + y.float().unsqueeze(-1)

    sup_ds = TensorDataset(x, y, z)
    unsup_ds = TensorDataset(x, z)
    sup_loader = DataLoader(sup_ds, batch_size=cfg.batch_size)
    unsup_loader = DataLoader(unsup_ds, batch_size=cfg.batch_size)
    return sup_loader, unsup_loader
