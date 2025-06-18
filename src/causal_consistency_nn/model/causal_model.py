"""Simple model combining the backbone and conditional heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import one_hot

from .backbone import Backbone
from .heads import XgivenYZ, YgivenXZ, ZgivenXY


class CausalModel(nn.Module):
    """Neural network enforcing causal consistency."""

    def __init__(
        self,
        backbone: Backbone,
        head_z: ZgivenXY,
        head_y: YgivenXZ,
        head_x: XgivenYZ,
        y_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_z = head_z
        self.head_y = head_y
        self.head_x = head_x
        self.y_dim = y_dim

    def head_z_given_xy(self, x: torch.Tensor, y: torch.Tensor):
        h = self.backbone(x)
        y_oh = one_hot(y, num_classes=self.y_dim).float()
        return self.head_z(h, y_oh).mean

    def head_y_given_xz(self, x: torch.Tensor, z: torch.Tensor):
        h = self.backbone(x)
        return self.head_y(h, z).logits

    def head_x_given_yz(self, y: torch.Tensor, z: torch.Tensor):
        h = self.backbone(z)
        y_oh = one_hot(y, num_classes=self.y_dim).float()
        return self.head_x(h, y_oh).mean
