"""Semi-supervised EM training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EMConfig:
    """Configuration for EM training."""

    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    beta: float = 1.0
    tau: float = 1.0
    lr: float = 1e-3
    epochs: int = 10
    pretrain_epochs: int = 0


def _supervised_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: EMConfig,
    mse: nn.Module,
    ce: nn.Module,
) -> torch.Tensor:
    x, y, z = batch
    z_pred = model.head_z_given_xy(x, y)
    y_logits = model.head_y_given_xz(x, z)
    x_pred = model.head_x_given_yz(y, z)

    loss_z = mse(z_pred, z)
    loss_y = ce(y_logits, y)
    loss_x = mse(x_pred, x)
    loss = config.lambda1 * loss_z + config.lambda2 * loss_y + config.lambda3 * loss_x
    return loss


def _unsupervised_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    config: EMConfig,
    mse: nn.Module,
) -> torch.Tensor:
    x, z = batch
    y_logits = model.head_y_given_xz(x, z)
    y_probs = F.softmax(y_logits / config.tau, dim=-1)
    y_pseudo = torch.argmax(y_probs, dim=-1)
    z_pred = model.head_z_given_xy(x, y_pseudo)
    x_pred = model.head_x_given_yz(y_pseudo, z)
    loss_z = mse(z_pred, z)
    loss_x = mse(x_pred, x)
    loss = config.beta * (config.lambda1 * loss_z + config.lambda3 * loss_x)
    return loss


def train_em(
    model: nn.Module,
    supervised_loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    unsupervised_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
    config: Optional[EMConfig] = None,
) -> None:
    """Train ``model`` using a simple EM loop."""
    if config is None:
        config = EMConfig()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        for batch in supervised_loader:
            optimizer.zero_grad()
            loss = _supervised_step(model, batch, config, mse, ce)
            loss.backward()
            optimizer.step()

        if unsupervised_loader is not None and epoch >= config.pretrain_epochs:
            for batch in unsupervised_loader:
                optimizer.zero_grad()
                loss = _unsupervised_step(model, batch, config, mse)
                loss.backward()
                optimizer.step()
