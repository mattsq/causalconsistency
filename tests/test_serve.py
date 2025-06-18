from __future__ import annotations

import torch

from causal_consistency_nn import train
from causal_consistency_nn.config import Settings
from causal_consistency_nn.data import get_synth_dataloaders
from causal_consistency_nn.serve import counterfactual_z, impute_y, predict_z


def test_serving_helpers() -> None:
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

    pred = predict_z(model, x_ex, y_ex)
    cf = counterfactual_z(model, x_ex, 1 - y_ex)
    post = impute_y(model, x_ex, z_ex)

    assert pred.shape == z_ex.shape
    assert cf.shape == z_ex.shape
    assert post.shape[0] == y_ex.shape[0]
    assert torch.allclose(post.sum(-1), torch.ones_like(post[:, 0]), atol=1e-5)
