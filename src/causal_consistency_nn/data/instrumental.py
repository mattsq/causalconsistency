from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import SyntheticDataConfig


def generate_instrumental(
    cfg: SyntheticDataConfig, seed: int | None = None
) -> TensorDataset:
    """Generate data with an instrumental variable ``W -> Y``.

    W ~ N(0,1)
    X ~ N(0,1)
    Y | W,X ~ Categorical(logits=j*(X+cfg.w_y_strength*W))
    Z | X,Y = X + Y + eps,  eps ~ N(0, noise_std)
    Missingness in Y follows ``cfg.missing_y_prob``.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    w = torch.randn(cfg.n_samples, cfg.w_dim, generator=g)
    x = torch.randn(cfg.n_samples, 1, generator=g)

    coeffs = torch.arange(cfg.num_classes, dtype=x.dtype, device=x.device)
    logits = (
        x.squeeze(-1)[:, None] + cfg.w_y_strength * w.squeeze(-1)[:, None]
    ) * coeffs
    probs = torch.softmax(logits, dim=-1)
    y = torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1)

    noise = torch.randn(cfg.n_samples, 1, generator=g) * cfg.noise_std
    z = x + y.float().unsqueeze(-1) + noise

    mask = torch.rand(cfg.n_samples, generator=g) > cfg.missing_y_prob
    mask = mask.to(torch.bool)

    return TensorDataset(w, x, y, z, mask)


def get_instrumental_dataloaders(
    cfg: SyntheticDataConfig, batch_size: int, seed: int | None = None
) -> Tuple[DataLoader, DataLoader]:
    """Return loaders for the instrumental dataset."""
    dataset = generate_instrumental(cfg, seed)
    w, x, y, z, mask = dataset.tensors

    sup_ds = TensorDataset(w[mask], x[mask], y[mask], z[mask])
    unsup_ds = TensorDataset(w[~mask], x[~mask], z[~mask])

    sup_loader = DataLoader(sup_ds, batch_size=batch_size, shuffle=True)
    if len(unsup_ds) > 0:
        unsup_loader = DataLoader(unsup_ds, batch_size=batch_size, shuffle=True)
    else:
        unsup_loader = DataLoader(unsup_ds, batch_size=batch_size, shuffle=False)
    return sup_loader, unsup_loader


__all__ = ["generate_instrumental", "get_instrumental_dataloaders"]
