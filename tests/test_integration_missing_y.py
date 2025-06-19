from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from causal_consistency_nn.config import ModelConfig, SyntheticDataConfig
from causal_consistency_nn.data.synthetic import (
    get_synth_dataloaders_mar,
    get_synth_dataloaders_mnar,
)
from causal_consistency_nn.train import ConsistencyModel
from causal_consistency_nn.model.semi_loop import EMConfig, train_em


@pytest.mark.parametrize(
    "loader_fn", [get_synth_dataloaders_mar, get_synth_dataloaders_mnar]
)
def test_loader_output_types(loader_fn) -> None:
    cfg = SyntheticDataConfig(n_samples=20, noise_std=0.1, missing_y_prob=0.5)
    sup, unsup = loader_fn(cfg, batch_size=4, seed=0)
    bx = next(iter(sup))
    bx_uns = next(iter(unsup))
    assert isinstance(sup, DataLoader)
    assert isinstance(unsup, DataLoader)
    assert isinstance(bx[0], torch.Tensor) and len(bx) == 3
    assert isinstance(bx_uns[0], torch.Tensor) and len(bx_uns) == 2


@pytest.mark.parametrize(
    "loader_fn", [get_synth_dataloaders_mar, get_synth_dataloaders_mnar]
)
def test_loss_decreases(loader_fn) -> None:
    cfg = SyntheticDataConfig(n_samples=50, noise_std=0.1, missing_y_prob=0.5)
    sup_loader, unsup_loader = loader_fn(cfg, batch_size=10, seed=0)

    x_ex, y_ex, z_ex = next(iter(sup_loader))
    model = ConsistencyModel(
        x_ex.shape[1], int(y_ex.max()) + 1, z_ex.shape[1], ModelConfig()
    )
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    def eval_loss() -> float:
        total = 0.0
        for x, y, z in sup_loader:
            with torch.no_grad():
                loss = mse(model.head_z_given_xy(x, y), z)
                loss += ce(model.head_y_given_xz(x, z), y)
                loss += mse(model.head_x_given_yz(y, z), x)
                total += loss.item()
        return total

    before = eval_loss()
    train_em(model, sup_loader, unsup_loader, EMConfig(epochs=3, lr=0.01))
    after = eval_loss()
    assert after < before
