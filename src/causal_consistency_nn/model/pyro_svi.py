from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import torch
from torch import nn

from .semi_loop import EMConfig


@dataclass
class SVIConfig(EMConfig):
    """Configuration for Pyro SVI training."""


def _model_supervised(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    cfg: SVIConfig,
) -> None:
    if len(batch) == 4:
        w, x, y, z = batch
        w_dist = model.head_w(model.backbone(x))
    else:
        x, y, z = batch
        w = None
    with pyro.plate("batch", x.shape[0]):
        h = model.backbone(x)
        if w is not None:
            with pyro.poutine.scale(scale=cfg.lambda_w):
                pyro.sample("w", w_dist.to_event(1), obs=w)
        z_dist = model.head_z(h, model._onehot(y))
        with pyro.poutine.scale(scale=cfg.lambda1):
            pyro.sample("z", z_dist.to_event(1), obs=z)
        y_dist = model.head_y(h, z)
        with pyro.poutine.scale(scale=cfg.lambda2):
            pyro.sample("y", y_dist, obs=y)
        x_dist = model.head_x(model.backbone(z), model._onehot(y))
        with pyro.poutine.scale(scale=cfg.lambda3):
            pyro.sample("x", x_dist.to_event(1), obs=x)


def _model_unsupervised(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    cfg: SVIConfig,
) -> None:
    if len(batch) == 3:
        w, x, z = batch
        loss_w_scale = cfg.lambda_w * cfg.beta
    else:
        x, z = batch
        w = None
        loss_w_scale = 0.0
    with pyro.plate("batch", x.shape[0]):
        h = model.backbone(x)
        if w is not None:
            with pyro.poutine.scale(scale=loss_w_scale):
                pyro.sample("w", model.head_w(h).to_event(1), obs=w)
        with pyro.poutine.scale(scale=0.0):
            y = pyro.sample("y", model.head_y(h, z))
        z_dist = model.head_z(h, model._onehot(y))
        with pyro.poutine.scale(scale=cfg.beta * cfg.lambda1):
            pyro.sample("z", z_dist.to_event(1), obs=z)
        x_dist = model.head_x(model.backbone(z), model._onehot(y))
        with pyro.poutine.scale(scale=cfg.beta * cfg.lambda3):
            pyro.sample("x", x_dist.to_event(1), obs=x)


def _guide_unsupervised(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    cfg: SVIConfig,
) -> None:
    if len(batch) == 3:
        _w, x, z = batch
    else:
        x, z = batch
    with pyro.plate("batch", x.shape[0]):
        h = model.backbone(x)
        pyro.sample("y", model.head_y(h, z))


def train_svi(
    model: nn.Module,
    supervised_loader: Iterable[Tuple[torch.Tensor, ...]],
    unsupervised_loader: Optional[Iterable[Tuple[torch.Tensor, ...]]],
    config: Optional[SVIConfig] = None,
    device: torch.device | str = "cpu",
) -> None:
    """Train ``model`` using Pyro SVI."""
    if config is None:
        config = SVIConfig()

    torch_device = torch.device(device)
    model.to(torch_device)

    pyro.clear_param_store()
    optim = Adam({"lr": config.lr})

    guide_sup = AutoNormal(lambda batch: _model_supervised(model, batch, config))
    svi_sup = SVI(
        lambda batch: _model_supervised(model, batch, config),
        guide_sup,
        optim,
        loss=Trace_ELBO(),
    )
    svi_unsup = SVI(
        lambda batch: _model_unsupervised(model, batch, config),
        lambda batch: _guide_unsupervised(model, batch, config),
        optim,
        loss=Trace_ELBO(),
    )

    for epoch in range(config.epochs):
        for batch in supervised_loader:
            batch = tuple(t.to(torch_device) for t in batch)
            svi_sup.step(batch)
        if unsupervised_loader is not None and epoch >= config.pretrain_epochs:
            for batch in unsupervised_loader:
                batch = tuple(t.to(torch_device) for t in batch)
                svi_unsup.step(batch)
