from __future__ import annotations

from fastapi.testclient import TestClient
import torch

from causal_consistency_nn import train
from causal_consistency_nn.config import Settings
from causal_consistency_nn.data import get_synth_dataloaders
from causal_consistency_nn.fastapi_app import create_app


def test_fastapi_endpoints() -> None:
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

    app = create_app(model)
    client = TestClient(app)

    resp = client.post(
        "/predict_z",
        json={"x": x_ex.tolist(), "y": y_ex.tolist()},
    )
    assert resp.status_code == 200
    assert torch.tensor(resp.json()["z"]).shape == z_ex.shape

    resp = client.post(
        "/counterfactual_z",
        json={"x": x_ex.tolist(), "y_prime": (1 - y_ex).tolist()},
    )
    assert resp.status_code == 200
    assert torch.tensor(resp.json()["z"]).shape == z_ex.shape

    resp = client.post(
        "/impute_y",
        json={"x": x_ex.tolist(), "z": z_ex.tolist()},
    )
    assert resp.status_code == 200
    prob = torch.tensor(resp.json()["y_prob"])
    assert prob.shape[0] == y_ex.shape[0]
    assert torch.allclose(prob.sum(-1), torch.ones_like(prob[:, 0]), atol=1e-5)
