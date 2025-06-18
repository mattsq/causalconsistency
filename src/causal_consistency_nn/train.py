"""Training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import Settings


def main(argv: list[str] | None = None) -> None:
    """Parse configuration and launch training (placeholder)."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
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
    print("Training placeholder")


if __name__ == "__main__":
    main()
