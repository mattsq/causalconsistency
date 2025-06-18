import torch
from torch.distributions import Normal, Categorical

from causal_consistency_nn.model.losses import (
    nll_gaussian,
    cross_entropy,
    entropy_categorical,
)


def test_nll_gaussian() -> None:
    dist = Normal(torch.tensor(0.0), torch.tensor(1.0))
    target = torch.tensor(0.0)
    loss = nll_gaussian(dist, target)
    expected = -dist.log_prob(target)
    assert torch.isclose(loss, expected)


def test_cross_entropy() -> None:
    dist = Categorical(probs=torch.tensor([0.1, 0.9]))
    target = torch.tensor(1)
    loss = cross_entropy(dist, target)
    expected = -dist.log_prob(target)
    assert torch.isclose(loss, expected)


def test_entropy_categorical() -> None:
    dist = Categorical(probs=torch.tensor([0.1, 0.9]))
    ent = entropy_categorical(dist)
    expected = dist.entropy()
    assert torch.isclose(ent, expected)
