from __future__ import annotations

import torch

from causal_consistency_nn.config import SyntheticDataConfig
from causal_consistency_nn.data.synthetic import (
    generate_synthetic,
    get_synth_dataloaders,
)

import pytest


@pytest.mark.parametrize("num_classes", [2, 3])
def test_generate_synthetic_shapes(num_classes: int) -> None:
    cfg = SyntheticDataConfig(
        n_samples=10, noise_std=0.1, missing_y_prob=0.2, num_classes=num_classes
    )
    ds = generate_synthetic(cfg, seed=0)
    x, y, z, m = ds.tensors
    assert x.shape == (10, 1)
    assert y.shape == (10,)
    assert z.shape == (10, 1)
    assert m.shape == (10,)


def test_generate_synthetic_reproducible() -> None:
    cfg = SyntheticDataConfig(
        n_samples=5, noise_std=0.0, missing_y_prob=0.0, num_classes=3
    )
    ds1 = generate_synthetic(cfg, seed=42)
    ds2 = generate_synthetic(cfg, seed=42)
    for t1, t2 in zip(ds1.tensors, ds2.tensors):
        assert torch.equal(t1, t2)


def test_get_synth_dataloaders() -> None:
    cfg = SyntheticDataConfig(
        n_samples=20, noise_std=0.1, missing_y_prob=0.5, num_classes=3
    )
    sup, unsup = get_synth_dataloaders(cfg, batch_size=4, seed=0)
    bx = next(iter(sup))
    assert len(bx) == 3
    bx_uns = next(iter(unsup))
    assert len(bx_uns) == 2
