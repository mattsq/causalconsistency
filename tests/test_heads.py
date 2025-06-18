import torch
from torch.nn.functional import one_hot

from causal_consistency_nn.model.heads import (
    XgivenYZ,
    XgivenYZConfig,
    YgivenXZ,
    YgivenXZConfig,
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

    dist_z = z_head(h, y_oh)
    assert dist_z.mean.shape == (batch, z_dim)

    dist_y = y_head(h, z)
    assert dist_y.logits.shape == (batch, y_dim)

    dist_x = x_head(h, y_oh)
    assert dist_x.mean.shape == (batch, x_dim)

    loss = (
        dist_z.log_prob(z).sum()
        + dist_y.log_prob(y).sum()
        + dist_x.log_prob(torch.randn(batch, x_dim)).sum()
    )
    loss.backward()

    for param in (
        list(z_head.parameters())
        + list(y_head.parameters())
        + list(x_head.parameters())
    ):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
