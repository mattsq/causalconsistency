from __future__ import annotations

from pathlib import Path

from torch import nn

from causal_consistency_nn import train
from causal_consistency_nn.config import Settings
from causal_consistency_nn.data import get_synth_dataloaders


def test_train_end_to_end(tmp_path: Path) -> None:
    settings = Settings()
    settings.train.epochs = 2
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

    before = eval_loss()
    train.train_em(model, sup_loader, unsup_loader, train.EMConfig(epochs=2, lr=0.01))
    after = eval_loss()

    assert after < before

    out_dir = tmp_path / "out"
    train.run_training(settings, out_dir)
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "config.yaml").exists()
