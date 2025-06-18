from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    hidden_dim: int = 64
    num_layers: int = 2


@dataclass
class LossWeights:
    """Weighting of individual loss terms."""

    z_yx: float = 1.0
    y_xz: float = 1.0
    x_yz: float = 1.0
    unsup: float = 0.0


@dataclass
class TrainingConfig:
    """General training parameters."""

    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"


@dataclass
class SyntheticDataConfig:
    """Parameters controlling synthetic data generation."""

    n_samples: int = 1000
    noise_std: float = 0.1
    missing_y_prob: float = 0.0


class Settings(BaseSettings):
    """Global configuration loaded from environment variables or a YAML file."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossWeights = Field(default_factory=LossWeights)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    data: SyntheticDataConfig = Field(default_factory=SyntheticDataConfig)
    config_path: str | None = None

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load configuration from a YAML file, allowing env overrides."""
        data: dict[str, Any] = {}
        with Path(path).open("r") as handle:
            data = yaml.safe_load(handle) or {}

        env_map = {
            "model": ModelConfig,
            "loss": LossWeights,
            "train": TrainingConfig,
            "data": SyntheticDataConfig,
        }
        for section, dc in env_map.items():
            if section in data:
                for field in list(dc.__annotations__):
                    env_var = f"{section.upper()}__{field.upper()}"
                    if env_var in os.environ:
                        data[section].pop(field, None)
                if not data[section]:
                    data.pop(section)

        return cls(**data, config_path=str(path))
