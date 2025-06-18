"""Simple evaluation CLI computing causal metrics."""

from __future__ import annotations

import argparse

import torch

from .metrics import average_treatment_effect, log_likelihood
from .serve import predict_z


def _create_synth(
    n: int = 100, noise: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(n, 1)
    y = (x.squeeze() > 0).long()
    z = x + y.float().unsqueeze(-1) + torch.randn_like(x) * noise
    return x, y, z


class _OracleModel:
    def __init__(self, noise: float = 0.1) -> None:
        self.noise = noise

    def head_z_given_xy(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.distributions.Normal:
        mu = x + y.float().unsqueeze(-1)
        sigma = torch.full_like(mu, self.noise)
        return torch.distributions.Normal(mu, sigma)

    def head_y_given_xz(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> torch.distributions.Categorical:
        logits = (
            torch.stack([-(z - x).squeeze(), (z - x).squeeze()], dim=-1) / self.noise
        )
        return torch.distributions.Categorical(logits=logits)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.parse_args(argv)

    x, y, z = _create_synth()
    model = _OracleModel()

    ate_val = average_treatment_effect(model, x)
    ll = log_likelihood(model.head_z_given_xy(x, y), z)

    preds = predict_z(model, x, y)
    print(f"ATE: {ate_val:.3f}, loglik_z: {ll:.3f}, pred_shape: {preds.shape}")


if __name__ == "__main__":
    main()
