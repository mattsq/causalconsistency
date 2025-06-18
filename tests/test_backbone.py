import torch

from causal_consistency_nn.model.backbone import Backbone, BackboneConfig


def test_backbone_forward() -> None:
    cfg = BackboneConfig(in_dims=4, hidden=(8, 6), activation="relu", dropout=0.0)
    model = Backbone(cfg)
    x = torch.randn(2, cfg.in_dims)
    out = model(x)
    assert out.shape == (2, 6)
