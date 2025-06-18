"""Public model components."""

from __future__ import annotations

from .backbone import Backbone, BackboneConfig
from .heads import (
    XgivenYZ,
    XgivenYZConfig,
    YgivenXZ,
    YgivenXZConfig,
    ZgivenXY,
    ZgivenXYConfig,
)
from .semi_loop import EMConfig, train_em

__all__ = [
    "Backbone",
    "BackboneConfig",
    "ZgivenXY",
    "ZgivenXYConfig",
    "YgivenXZ",
    "YgivenXZConfig",
    "XgivenYZ",
    "XgivenYZConfig",
    "EMConfig",
    "train_em",
]


def build_backbone(config: BackboneConfig) -> Backbone:
    """Convenience wrapper returning a :class:`Backbone`."""
    return Backbone(config)
