from __future__ import annotations

import torch

from causal_consistency_nn.metrics import (
    average_treatment_effect,
    log_likelihood_normal,
)
from causal_consistency_nn.config import SyntheticDataConfig
from causal_consistency_nn.data.synthetic import generate_synthetic


def test_average_treatment_effect_synth() -> None:
    cfg = SyntheticDataConfig(
        n_samples=500, noise_std=0.1, missing_y_prob=0.0, num_classes=2
    )
    ds = generate_synthetic(cfg, seed=0)
    x, _, _, _ = ds.tensors
    z_treat = (x + 1).squeeze()
    z_control = x.squeeze()
    y_long = torch.cat(
        [
            torch.ones_like(z_treat, dtype=torch.long),
            torch.zeros_like(z_control, dtype=torch.long),
        ]
    )
    z_all = torch.cat([z_treat, z_control])
    ate = average_treatment_effect(y_long, z_all)
    assert torch.isclose(ate, torch.tensor(1.0), atol=1e-6)


def test_log_likelihood_normal() -> None:
    mu = torch.zeros(4, 1)
    sigma = torch.ones(4, 1)
    target = torch.randn(4, 1)
    ll = log_likelihood_normal(mu, sigma, target)
    expected = torch.distributions.Normal(mu, sigma).log_prob(target).mean()
    assert torch.isclose(ll, expected)
