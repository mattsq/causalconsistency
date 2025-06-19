from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    hidden_dim: int = field(
        default=64,
        metadata={"help": "Number of features in each hidden layer of the heads."},
    )
    num_layers: int = field(
        default=2,
        metadata={"help": "Depth of the shared backbone in terms of hidden layers."},
    )


@dataclass
class LossWeights:
    """Weighting of individual loss terms."""

    z_yx: float = field(
        default=1.0,
        metadata={"help": "Weight for predicting Z from X and Y."},
    )
    y_xz: float = field(
        default=1.0,
        metadata={"help": "Weight for predicting Y from X and Z."},
    )
    x_yz: float = field(
        default=1.0,
        metadata={"help": "Weight for reconstructing X from Y and Z."},
    )
    unsup: float = field(
        default=0.0,
        metadata={"help": "Multiplier on the unsupervised objective."},
    )


@dataclass
class TrainingConfig:
    """General training parameters."""

    batch_size: int = field(
        default=32,
        metadata={"help": "Number of samples per optimisation step."},
    )
    epochs: int = field(
        default=10,
        metadata={"help": "Total training epochs to run."},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Initial learning rate for the optimiser."},
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Torch device identifier such as 'cpu' or 'cuda'."},
    )
    use_pyro: bool = field(
        default=False,
        metadata={"help": "Enable Pyro-based layers for probabilistic training."},
    )


@dataclass
class SyntheticDataConfig:
    """Parameters controlling synthetic data generation."""

    n_samples: int = field(
        default=1000,
        metadata={"help": "Number of examples to generate in the toy dataset."},
    )
    noise_std: float = field(
        default=0.1,
        metadata={"help": "Standard deviation of Gaussian noise added to Z."},
    )
    missing_y_prob: float = field(
        default=0.0,
        metadata={"help": "Probability that Y is masked during training."},
    )


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
