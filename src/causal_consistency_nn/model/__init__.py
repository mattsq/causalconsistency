"""Public model API."""

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
from .pyro_model import PyroConsistencyModel
from .pyro_svi import SVIConfig, train_svi

try:  # optional dependency
    from .lightning_loop import (
        LightningConfig,
        LightningConsistencyModule,
        train_lightning,
    )
except ModuleNotFoundError:  # pytorch_lightning missing
    LightningConfig = None  # type: ignore[assignment]
    LightningConsistencyModule = None  # type: ignore[assignment]
    train_lightning = None  # type: ignore[assignment]

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
    "PyroConsistencyModel",
    "SVIConfig",
    "train_svi",
]

if LightningConfig is not None:
    __all__ += [
        "LightningConfig",
        "LightningConsistencyModule",
        "train_lightning",
    ]
