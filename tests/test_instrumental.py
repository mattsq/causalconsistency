from __future__ import annotations

from torch.utils.data import DataLoader

from causal_consistency_nn.config import SyntheticDataConfig
from causal_consistency_nn.data.instrumental import (
    generate_instrumental,
    get_instrumental_dataloaders,
)


def test_generate_instrumental_shapes() -> None:
    cfg = SyntheticDataConfig(n_samples=10, instrumental=True)
    ds = generate_instrumental(cfg, seed=0)
    w, x, y, z, m = ds.tensors
    assert w.shape == (10, cfg.w_dim)
    assert x.shape == (10, 1)
    assert y.shape == (10,)
    assert z.shape == (10, 1)
    assert m.shape == (10,)


def test_instrumental_dataloaders() -> None:
    cfg = SyntheticDataConfig(n_samples=20, instrumental=True)
    sup, unsup = get_instrumental_dataloaders(cfg, batch_size=4, seed=0)
    assert isinstance(sup, DataLoader)
    assert isinstance(unsup, DataLoader)
    bw = next(iter(sup))
    assert len(bw) == 4
    if len(unsup) > 0:
        bu = next(iter(unsup))
        assert len(bu) == 3
