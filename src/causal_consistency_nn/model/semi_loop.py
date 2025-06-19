"""Semi-supervised EM training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EMConfig:
    """Configuration for EM training.

    ``tau`` controls the softness of the pseudo-label distribution and
    weights the entropy regularisation in the unsupervised step.
    """

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

    n_classes = y_probs.shape[1]
    loss_z = []
    loss_x = []
    for cls in range(n_classes):
        y_cls = torch.full((x.shape[0],), cls, dtype=torch.long, device=x.device)
        z_pred = model.head_z_given_xy(x, y_cls)
        x_pred = model.head_x_given_yz(y_cls, z)
        loss_z.append(F.mse_loss(z_pred, z, reduction="none").mean(dim=1))
        loss_x.append(F.mse_loss(x_pred, x, reduction="none").mean(dim=1))

    loss_z = torch.stack(loss_z, dim=-1)
    loss_x = torch.stack(loss_x, dim=-1)
    exp_loss_z = (loss_z * y_probs).sum(dim=-1).mean()
    exp_loss_x = (loss_x * y_probs).sum(dim=-1).mean()
    entropy = -(y_probs * torch.log(y_probs + 1e-8)).sum(dim=-1).mean()

    loss = (
        config.beta * (config.lambda1 * exp_loss_z + config.lambda3 * exp_loss_x)
        - config.tau * entropy
    )
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
        sup_total = 0.0
        sup_batches = 0
        for batch in supervised_loader:
            optimizer.zero_grad()
            loss = _supervised_step(model, batch, config, mse, ce)
            loss.backward()
            optimizer.step()
            sup_total += loss.item()
            sup_batches += 1

        unsup_total = 0.0
        unsup_batches = 0
        if unsupervised_loader is not None and epoch >= config.pretrain_epochs:
            for batch in unsupervised_loader:
                optimizer.zero_grad()
                loss = _unsupervised_step(model, batch, config, mse)
                loss.backward()
                optimizer.step()
                unsup_total += loss.item()
                unsup_batches += 1

        avg_sup = sup_total / max(1, sup_batches)
        avg_unsup = unsup_total / max(1, unsup_batches)
        from causal_consistency_nn.utils.logging import log

        log(
            f"Epoch {epoch}: supervised_loss={avg_sup:.4f}, unsupervised_loss={avg_unsup:.4f}"
        )
