"""Evaluation utilities computing causal metrics."""

from __future__ import annotations


from pathlib import Path
import argparse
import yaml
import torch

from .config import Settings
from .data import get_synth_dataloaders, get_instrumental_dataloaders
from .train import ConsistencyModel
from .metrics import dataset_log_likelihood


def load_model(model_path: Path, settings: Settings) -> ConsistencyModel:
    """Load ``ConsistencyModel`` from ``model_path``."""
    if settings.data.instrumental:
        sup_loader, _ = get_instrumental_dataloaders(
            settings.data, batch_size=settings.train.batch_size, seed=0
        )
        _, x_ex, y_ex, z_ex = next(iter(sup_loader))
    else:
        sup_loader, _ = get_synth_dataloaders(
            settings.data, batch_size=settings.train.batch_size, seed=0
        )
        x_ex, y_ex, z_ex = next(iter(sup_loader))
    model = ConsistencyModel(
        x_ex.shape[1], int(y_ex.max().item()) + 1, z_ex.shape[1], settings.model
    )
    state = torch.load(model_path, map_location=settings.train.device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(model: ConsistencyModel, loader) -> dict[str, float]:
    """Return log likelihood and ATE estimates for ``model`` on ``loader``."""
    ll = dataset_log_likelihood(model, loader)
    xs: list[torch.Tensor] = []
    for batch in loader:
        xs.append(batch[0])
    x_all = torch.cat(xs)
    with torch.no_grad():
        z_treat = model.head_z_given_xy(x_all, torch.ones(len(x_all), dtype=torch.long))
        z_control = model.head_z_given_xy(
            x_all, torch.zeros(len(x_all), dtype=torch.long)
        )
    ate = (z_treat.mean() - z_control.mean()).item()
    return {"log_likelihood": ll, "ate": ate}


def main(argv: list[str] | None = None) -> None:
    """CLI for evaluation on synthetic data."""
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    settings = Settings.from_yaml(args.config)
    if settings.data.instrumental:
        sup_loader, _ = get_instrumental_dataloaders(
            settings.data, batch_size=settings.train.batch_size, seed=0
        )
    else:
        sup_loader, _ = get_synth_dataloaders(
            settings.data, batch_size=settings.train.batch_size, seed=0
        )
    model = load_model(args.model_path, settings)
    metrics = evaluate(model, sup_loader)
    print(yaml.safe_dump(metrics))


if __name__ == "__main__":
    main()
