import torch
from torch.nn.functional import one_hot

import pytest

from causal_consistency_nn.model.heads import (
    XgivenYZ,
    XgivenYZConfig,
    YgivenXZ,
    YgivenXZConfig,
    WgivenX,
    WgivenXConfig,
    ZgivenXY,
    ZgivenXYConfig,
)


def _make_onehot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return one_hot(y, num_classes=num_classes).float()


def test_heads_shapes_and_gradients() -> None:
    batch = 2
    h_dim = 16
    x_dim = 5
    y_dim = 3
    z_dim = 4

    h = torch.randn(batch, h_dim, requires_grad=True)
    y = torch.randint(0, y_dim, (batch,))
    y_oh = _make_onehot(y, y_dim)
    z = torch.randn(batch, z_dim)

    z_head = ZgivenXY(ZgivenXYConfig(h_dim=h_dim, y_dim=y_dim, z_dim=z_dim))
    y_head = YgivenXZ(YgivenXZConfig(h_dim=h_dim, z_dim=z_dim, y_dim=y_dim))
    x_head = XgivenYZ(XgivenYZConfig(h_dim=h_dim, y_dim=y_dim, x_dim=x_dim))
    w_head = WgivenX(WgivenXConfig(h_dim=h_dim, w_dim=2))

    dist_z = z_head(h, y_oh)
    assert dist_z.mean.shape == (batch, z_dim)

    dist_y = y_head(h, z)
    assert dist_y.logits.shape == (batch, y_dim)

    dist_x = x_head(h, y_oh)
    dist_w = w_head(h)
    assert dist_x.mean.shape == (batch, x_dim)
    assert dist_w.mean.shape == (batch, 2)

    loss = (
        dist_z.log_prob(z).sum()
        + dist_y.log_prob(y).sum()
        + dist_x.log_prob(torch.randn(batch, x_dim)).sum()
        + dist_w.log_prob(torch.randn(batch, 2)).sum()
    )
    loss.backward()

    for param in (
        list(z_head.parameters())
        + list(y_head.parameters())
        + list(x_head.parameters())
        + list(w_head.parameters())
    ):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_head_shape_validation() -> None:
    cfg_z = ZgivenXYConfig(h_dim=4, y_dim=2, z_dim=3)
    cfg_y = YgivenXZConfig(h_dim=4, z_dim=3, y_dim=2)
    cfg_x = XgivenYZConfig(h_dim=4, y_dim=2, x_dim=5)
    cfg_w = WgivenXConfig(h_dim=4, w_dim=1)

    z_head = ZgivenXY(cfg_z)
    y_head = YgivenXZ(cfg_y)
    x_head = XgivenYZ(cfg_x)
    w_head = WgivenX(cfg_w)

    h_bad = torch.randn(2, cfg_z.h_dim + 1)
    y_oh = torch.zeros(2, cfg_z.y_dim)
    with pytest.raises(ValueError):
        z_head(h_bad, y_oh)

    z_bad = torch.randn(3, cfg_y.z_dim)
    h_ok = torch.randn(2, cfg_y.h_dim)
    with pytest.raises(ValueError):
        y_head(h_ok, z_bad)

    y_oh_bad = torch.zeros(2, cfg_x.y_dim + 1)
    with pytest.raises(ValueError):
        x_head(h_ok, y_oh_bad)

    with pytest.raises(ValueError):
        w_head(h_bad)
