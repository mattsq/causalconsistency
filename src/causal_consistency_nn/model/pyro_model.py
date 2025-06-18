from __future__ import annotations

from pyro.nn import PyroModule
import torch
import torch.nn.functional as F

from .backbone import Backbone, BackboneConfig
from .heads import ZgivenXYConfig, YgivenXZConfig, XgivenYZConfig
from .pyro_zgivenxy import PyroZgivenXY
from .pyro_ygivenxz import PyroYgivenXZ
from .pyro_xgivenyz import PyroXgivenYZ
from ..config import ModelConfig


class PyroConsistencyModel(PyroModule):
    """Consistency model using Pyro heads."""

    def __init__(self, x_dim: int, y_dim: int, z_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = [cfg.hidden_dim] * cfg.num_layers
        self.backbone = PyroModule[Backbone](
            BackboneConfig(in_dims=x_dim, hidden=hidden)
        )
        h_dim = self.backbone.output_dim
        self.head_z = PyroZgivenXY(
            ZgivenXYConfig(h_dim=h_dim, y_dim=y_dim, z_dim=z_dim)
        )
        self.head_y = PyroYgivenXZ(
            YgivenXZConfig(h_dim=h_dim, z_dim=z_dim, y_dim=y_dim)
        )
        self.head_x = PyroXgivenYZ(
            XgivenYZConfig(h_dim=h_dim, y_dim=y_dim, x_dim=x_dim)
        )
        self.y_dim = y_dim

    def _onehot(self, y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y, num_classes=self.y_dim).float()

    def head_z_given_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head_z(h, self._onehot(y)).mean

    def head_y_given_xz(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head_y(h, z).logits

    def head_x_given_yz(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.backbone(z)
        return self.head_x(h, self._onehot(y)).mean


__all__ = ["PyroConsistencyModel"]
