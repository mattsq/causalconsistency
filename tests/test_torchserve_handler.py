from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from causal_consistency_nn import train
from causal_consistency_nn.config import Settings
from causal_consistency_nn.data import get_synth_dataloaders
from causal_consistency_nn.torchserve_handler import handle


class DummyCtx:
    def __init__(self, model_dir: Path) -> None:
        self.system_properties = {"model_dir": str(model_dir)}


def test_handle_shapes(tmp_path: Path) -> None:
    settings = Settings()
    settings.train.epochs = 1
    sup, unsup = get_synth_dataloaders(
        settings.data, batch_size=settings.train.batch_size, seed=0
    )
    x_ex, y_ex, z_ex = next(iter(sup))
    model = train.ConsistencyModel(
        x_ex.shape[1], int(y_ex.max().item()) + 1, z_ex.shape[1], settings.model
    )
    train.train_em(model, sup, unsup, train.EMConfig(epochs=1))

    torch.save(model.state_dict(), tmp_path / "model.pt")
    with (tmp_path / "config.yaml").open("w") as fh:
        yaml.safe_dump(settings.model_dump(exclude={"config_path"}), fh)

    ctx = DummyCtx(tmp_path)

    req = [
        {
            "body": json.dumps(
                {"action": "predict_z", "x": x_ex.tolist(), "y": y_ex.tolist()}
            )
        }
    ]
    out = handle(req, ctx)
    assert torch.tensor(out).shape == z_ex.shape

    req = [
        {
            "body": json.dumps(
                {"action": "impute_y", "x": x_ex.tolist(), "z": z_ex.tolist()}
            )
        }
    ]
    out = handle(req, ctx)
    assert torch.tensor(out).shape[0] == y_ex.shape[0]

    req = [
        {
            "body": json.dumps(
                {
                    "action": "counterfactual_z",
                    "x": x_ex.tolist(),
                    "y_prime": (1 - y_ex).tolist(),
                }
            )
        }
    ]
    out = handle(req, ctx)
    assert torch.tensor(out).shape == z_ex.shape
