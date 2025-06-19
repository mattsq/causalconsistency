from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pytorch_lightning")

import torch
from torch import nn

from causal_consistency_nn import train
from causal_consistency_nn.config import Settings
from causal_consistency_nn.data import get_synth_dataloaders
from causal_consistency_nn.model.lightning_loop import train_lightning, LightningConfig


def test_lightning_end_to_end(tmp_path: Path) -> None:
    settings = Settings()
    settings.train.use_lightning = True
    settings.train.epochs = 4
    settings.train.learning_rate = 0.01
    sup_loader, unsup_loader = get_synth_dataloaders(
        settings.data, batch_size=settings.train.batch_size, seed=0
    )

    x_ex, y_ex, z_ex = next(iter(sup_loader))
    model = train.ConsistencyModel(
        x_ex.shape[1], int(y_ex.max().item()) + 1, z_ex.shape[1], settings.model
    )

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    def eval_loss() -> float:
        total = 0.0
        for x, y, z in sup_loader:
            total += (
                mse(model.head_z_given_xy(x, y), z)
                + ce(model.head_y_given_xz(x, z), y)
                + mse(model.head_x_given_yz(y, z), x)
            ).item()
        return total

    ckpt_dir = tmp_path / "ckpt"
    before_params = [p.clone() for p in model.parameters()]
    train_lightning(
        model,
        sup_loader,
        unsup_loader,
        LightningConfig(
            epochs=4,
            lr=0.01,
            checkpoint_dir=ckpt_dir,
            early_stopping_patience=1,
        ),
    )
    after_params = list(model.parameters())

    changed = any(not torch.allclose(b, a) for b, a in zip(before_params, after_params))
    assert changed

    assert any(ckpt_dir.glob("*.ckpt"))

    out_dir = tmp_path / "out"
    train.run_training(settings, out_dir)
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "config.yaml").exists()
