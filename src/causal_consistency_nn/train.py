"""Training entry point."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig, Settings
from .data import get_synth_dataloaders
from .model import (
    Backbone,
    BackboneConfig,
    EMConfig,
    PyroConsistencyModel,
    SVIConfig,
    XgivenYZ,
    XgivenYZConfig,
    YgivenXZ,
    YgivenXZConfig,
    ZgivenXY,
    ZgivenXYConfig,
    train_em,
    train_svi,
)


class ConsistencyModel(nn.Module):
    """Minimal model combining backbone and heads."""

    def __init__(self, x_dim: int, y_dim: int, z_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = [cfg.hidden_dim] * cfg.num_layers
        self.backbone = Backbone(BackboneConfig(in_dims=x_dim, hidden=hidden))
        h_dim = self.backbone.output_dim
        self.head_z = ZgivenXY(ZgivenXYConfig(h_dim=h_dim, y_dim=y_dim, z_dim=z_dim))
        self.head_y = YgivenXZ(YgivenXZConfig(h_dim=h_dim, z_dim=z_dim, y_dim=y_dim))
        self.head_x = XgivenYZ(XgivenYZConfig(h_dim=h_dim, y_dim=y_dim, x_dim=x_dim))
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


def run_training(settings: Settings, out_dir: Path) -> None:
    """Train a model using the provided ``settings`` and save outputs."""

    sup_loader, unsup_loader = get_synth_dataloaders(
        settings.data, batch_size=settings.train.batch_size, seed=0
    )

    x_example, y_example, z_example = next(iter(sup_loader))
    x_dim = x_example.shape[1]
    y_dim = int(y_example.max().item()) + 1
    z_dim = z_example.shape[1]

    if settings.train.use_pyro:
        model = PyroConsistencyModel(x_dim, y_dim, z_dim, settings.model)
        svi_cfg = SVIConfig(
            lambda1=settings.loss.z_yx,
            lambda2=settings.loss.y_xz,
            lambda3=settings.loss.x_yz,
            beta=settings.loss.unsup,
            lr=settings.train.learning_rate,
            epochs=settings.train.epochs,
        )
        train_svi(model, sup_loader, unsup_loader, svi_cfg)
    else:
        model = ConsistencyModel(x_dim, y_dim, z_dim, settings.model)
        em_cfg = EMConfig(
            lambda1=settings.loss.z_yx,
            lambda2=settings.loss.y_xz,
            lambda3=settings.loss.x_yz,
            beta=settings.loss.unsup,
            lr=settings.train.learning_rate,
            epochs=settings.train.epochs,
        )
        train_em(model, sup_loader, unsup_loader, em_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with (out_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(settings.model_dump(), handle)

    # Copy environment lockfile for reproducibility
    repo_root = Path(__file__).resolve().parents[2]
    lockfile = repo_root / "conda-lock.yml"
    if lockfile.exists():
        shutil.copy(lockfile, out_dir / "conda-lock.yml")


def main(argv: list[str] | None = None) -> None:
    """Parse configuration and launch training."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model-hidden-dim", type=int)
    parser.add_argument("--model-num-layers", type=int)
    parser.add_argument("--loss-z-yx", type=float)
    parser.add_argument("--loss-y-xz", type=float)
    parser.add_argument("--loss-x-yz", type=float)
    parser.add_argument("--loss-unsup", type=float)
    parser.add_argument("--use-pyro", action="store_true", help="Use Pyro SVI trainer")
    parser.add_argument("--out-dir", type=Path, default=Path("run"))
    args = parser.parse_args(argv)

    data: dict[str, object] = {}
    if args.config:
        with Path(args.config).open("r") as handle:
            data = yaml.safe_load(handle) or {}

    overrides: dict[str, dict[str, float | int | bool]] = {}
    if args.model_hidden_dim is not None or args.model_num_layers is not None:
        overrides["model"] = {}
        if args.model_hidden_dim is not None:
            overrides["model"]["hidden_dim"] = args.model_hidden_dim
        if args.model_num_layers is not None:
            overrides["model"]["num_layers"] = args.model_num_layers

    if (
        args.loss_z_yx is not None
        or args.loss_y_xz is not None
        or args.loss_x_yz is not None
        or args.loss_unsup is not None
    ):
        overrides["loss"] = {}
        if args.loss_z_yx is not None:
            overrides["loss"]["z_yx"] = args.loss_z_yx
        if args.loss_y_xz is not None:
            overrides["loss"]["y_xz"] = args.loss_y_xz
        if args.loss_x_yz is not None:
            overrides["loss"]["x_yz"] = args.loss_x_yz
        if args.loss_unsup is not None:
            overrides["loss"]["unsup"] = args.loss_unsup

    if args.use_pyro:
        overrides.setdefault("train", {})["use_pyro"] = True

    merged: dict[str, object] = {**data}
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value

    settings = Settings(**merged, config_path=args.config)

    print(f"Using config: {settings}")

    run_training(settings, args.out_dir)


if __name__ == "__main__":
    main()
