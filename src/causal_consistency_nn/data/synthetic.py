from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import SyntheticDataConfig


def generate_synthetic(
    cfg: SyntheticDataConfig, seed: int | None = None
) -> TensorDataset:
    """Generate a synthetic dataset following a simple SCM.

    X ~ N(0,1)
    Y | X ~ Categorical(logits=j*X) for j in 0..num_classes-1
    Z | X,Y = X + Y + eps,  eps ~ N(0, noise_std)

    Missingness in Y is governed by ``cfg.missing_y_prob``.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    x = torch.randn(cfg.n_samples, 1, generator=g)
    coeffs = torch.arange(cfg.num_classes, dtype=x.dtype, device=x.device)
    logits = x.squeeze(-1)[:, None] * coeffs
    probs = torch.softmax(logits, dim=-1)
    y = torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1)
    noise = torch.randn(cfg.n_samples, 1, generator=g) * cfg.noise_std
    z = x + y.float().unsqueeze(-1) + noise

    mask = torch.rand(cfg.n_samples, generator=g) > cfg.missing_y_prob
    mask = mask.to(torch.bool)

    return TensorDataset(x, y, z, mask)


def get_synth_dataloaders(
    cfg: SyntheticDataConfig, batch_size: int, seed: int | None = None
) -> Tuple[DataLoader, DataLoader]:
    """Return supervised and unsupervised dataloaders."""
    dataset = generate_synthetic(cfg, seed)
    x, y, z, mask = dataset.tensors

    sup_ds = TensorDataset(x[mask], y[mask], z[mask])
    unsup_ds = TensorDataset(x[~mask], z[~mask])

    sup_loader = DataLoader(sup_ds, batch_size=batch_size, shuffle=True)
    if len(unsup_ds) > 0:
        unsup_loader = DataLoader(unsup_ds, batch_size=batch_size, shuffle=True)
    else:
        unsup_loader = DataLoader(unsup_ds, batch_size=batch_size, shuffle=False)
    return sup_loader, unsup_loader


__all__ = ["generate_synthetic", "get_synth_dataloaders"]
