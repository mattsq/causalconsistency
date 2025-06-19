from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from causal_consistency_nn.config import ModelConfig, SyntheticDataConfig
from causal_consistency_nn.data.synthetic import (
    get_synth_dataloaders_mar,
    get_synth_dataloaders_mnar,
    generate_synthetic,
)
from causal_consistency_nn.train import ConsistencyModel
from causal_consistency_nn.model.semi_loop import EMConfig, train_em
from causal_consistency_nn.metrics import dataset_log_likelihood


@pytest.mark.parametrize(
    "loader_fn", [get_synth_dataloaders_mar, get_synth_dataloaders_mnar]
)
def test_semi_supervised_beats_supervised(loader_fn) -> None:
    cfg = SyntheticDataConfig(n_samples=400, noise_std=0.1, missing_y_prob=0.5)
    sup_loader, unsup_loader = loader_fn(cfg, batch_size=32, seed=0)

    x_ex, y_ex, z_ex = next(iter(sup_loader))
    model_sup = ConsistencyModel(
        x_ex.shape[1], int(y_ex.max()) + 1, z_ex.shape[1], ModelConfig()
    )
    model_semi = ConsistencyModel(
        x_ex.shape[1], int(y_ex.max()) + 1, z_ex.shape[1], ModelConfig()
    )

    train_em(model_sup, sup_loader, unsup_loader, EMConfig(beta=0.0, epochs=4, lr=0.01))
    train_em(
        model_semi, sup_loader, unsup_loader, EMConfig(beta=1.0, epochs=4, lr=0.01)
    )

    val_cfg = SyntheticDataConfig(n_samples=200, noise_std=0.1, missing_y_prob=0.0)
    val_ds = generate_synthetic(val_cfg, seed=1)
    x_val, y_val, z_val, _ = val_ds.tensors
    val_loader = DataLoader(TensorDataset(x_val, y_val, z_val), batch_size=64)

    ll_sup = dataset_log_likelihood(model_sup, val_loader)
    ll_semi = dataset_log_likelihood(model_semi, val_loader)
    assert ll_semi > ll_sup

    with torch.no_grad():
        z1_sup = model_sup.head_z_given_xy(
            x_val, torch.ones(len(x_val), dtype=torch.long)
        )
        z0_sup = model_sup.head_z_given_xy(
            x_val, torch.zeros(len(x_val), dtype=torch.long)
        )
        z1_semi = model_semi.head_z_given_xy(
            x_val, torch.ones(len(x_val), dtype=torch.long)
        )
        z0_semi = model_semi.head_z_given_xy(
            x_val, torch.zeros(len(x_val), dtype=torch.long)
        )

    ate_true = 1.0
    err_sup = abs((z1_sup - z0_sup).mean().item() - ate_true)
    err_semi = abs((z1_semi - z0_semi).mean().item() - ate_true)
    assert err_semi < err_sup
