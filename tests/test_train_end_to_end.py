from __future__ import annotations

from torch import nn

from causal_consistency_nn.config import Settings
from causal_consistency_nn.data import SynthConfig, make_synthetic_loaders
from causal_consistency_nn.train import run_training


def test_end_to_end_training(tmp_path) -> None:
    settings = Settings()
    settings.train.batch_size = 5
    settings.train.learning_rate = 0.01

    sup_loader, _ = make_synthetic_loaders(SynthConfig(n=40, batch_size=5))

    def eval_loss(model) -> float:
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        total = 0.0
        for x, y, z in sup_loader:
            z_pred = model.head_z_given_xy(x, y)
            y_logits = model.head_y_given_xz(x, z)
            x_pred = model.head_x_given_yz(y, z)
            loss = mse(z_pred, z) + ce(y_logits, y) + mse(x_pred, x)
            total += loss.item()
        return total

    settings.train.epochs = 0
    model_before = run_training(settings, tmp_path / "before")
    loss_before = eval_loss(model_before)

    settings.train.epochs = 3
    model_after = run_training(settings, tmp_path / "after")
    loss_after = eval_loss(model_after)

    assert loss_after < loss_before
