"""Training entry point for causal-consistency models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from .config import Settings
from .data import SynthConfig, make_synthetic_loaders
from .model import (
    Backbone,
    BackboneConfig,
    CausalModel,
    EMConfig,
    XgivenYZ,
    XgivenYZConfig,
    YgivenXZ,
    YgivenXZConfig,
    ZgivenXY,
    ZgivenXYConfig,
    train_em,
)


def run_training(settings: Settings, output_dir: Path) -> CausalModel:
    """Construct model, run EM training, and save artefacts."""
    sup_loader, unsup_loader = make_synthetic_loaders(
        SynthConfig(batch_size=settings.train.batch_size)
    )

    x_dim = 1
    z_dim = 1
    y_dim = 2
    hidden = (settings.model.hidden_dim,) * settings.model.num_layers

    backbone = Backbone(BackboneConfig(in_dims=x_dim, hidden=hidden))
    head_z = ZgivenXY(
        ZgivenXYConfig(h_dim=backbone.output_dim, y_dim=y_dim, z_dim=z_dim)
    )
    head_y = YgivenXZ(
        YgivenXZConfig(h_dim=backbone.output_dim, z_dim=z_dim, y_dim=y_dim)
    )
    head_x = XgivenYZ(
        XgivenYZConfig(h_dim=backbone.output_dim, y_dim=y_dim, x_dim=x_dim)
    )

    model = CausalModel(backbone, head_z, head_y, head_x, y_dim)

    em_cfg = EMConfig(
        lambda1=settings.loss.z_yx,
        lambda2=settings.loss.y_xz,
        lambda3=settings.loss.x_yz,
        beta=settings.loss.unsup,
        lr=settings.train.learning_rate,
        epochs=settings.train.epochs,
    )

    train_em(model, sup_loader, unsup_loader, em_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(settings.model_dump(), handle)
    return model


def main(argv: list[str] | None = None) -> None:
    """Parse configuration and launch training."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default="runs/default")
    parser.add_argument("--model-hidden-dim", type=int)
    parser.add_argument("--model-num-layers", type=int)
    parser.add_argument("--loss-z-yx", type=float)
    parser.add_argument("--loss-y-xz", type=float)
    parser.add_argument("--loss-x-yz", type=float)
    parser.add_argument("--loss-unsup", type=float)
    args = parser.parse_args(argv)

    data: dict[str, object] = {}
    if args.config:
        with Path(args.config).open("r") as handle:
            data = yaml.safe_load(handle) or {}

    overrides: dict[str, dict[str, float | int]] = {}
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

    merged: dict[str, object] = {**data}
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value

    settings = Settings(**merged, config_path=args.config)

    print(f"Using config: {settings}")

    run_training(settings, Path(args.output_dir))


if __name__ == "__main__":
    main()
