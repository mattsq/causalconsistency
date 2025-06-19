from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Sequence

import pytorch_lightning as pl
import torch
from torch import nn

from .semi_loop import EMConfig, _supervised_step, _unsupervised_step


@dataclass
class LightningConfig(EMConfig):
    """Configuration for Lightning training."""


class LightningConsistencyModule(pl.LightningModule):
    """Wraps ``ConsistencyModel`` for Lightning training."""

    def __init__(
        self,
        model: nn.Module,
        supervised_loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        unsupervised_loader: Optional[
            Iterable[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        config: Optional[LightningConfig] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.cfg = config or LightningConfig()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            loss = _supervised_step(self.model, batch, self.cfg, self.mse, self.ce)
            self.log("supervised_loss", loss, prog_bar=False, on_epoch=True)
        else:
            if self.current_epoch < self.cfg.pretrain_epochs:
                return None
            loss = _unsupervised_step(self.model, batch, self.cfg, self.mse)
            self.log("unsupervised_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def train_dataloader(self) -> Iterable | Sequence[Iterable]:
        """Return dataloaders for training."""
        if self.unsupervised_loader is None or len(self.unsupervised_loader) == 0:
            return self.supervised_loader
        return [self.supervised_loader, self.unsupervised_loader]


def train_lightning(
    model: nn.Module,
    supervised_loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    unsupervised_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
    config: Optional[LightningConfig] = None,
) -> None:
    """Train ``model`` using PyTorch Lightning."""
    cfg = config or LightningConfig()
    module = LightningConsistencyModule(
        model, supervised_loader, unsupervised_loader, cfg
    )
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module)


__all__ = ["LightningConfig", "LightningConsistencyModule", "train_lightning"]
