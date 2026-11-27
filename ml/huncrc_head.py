"""LightningModule for the HunCRC tile heads (linear or MLP on frozen embeddings).

Training matches the hand-written loop of 8b5955c bit for bit: AdamW over the
head's parameters, BCE with logits per tile (optionally class-weighted), and
dropout seeded from ``(seed, epoch)`` only, so a resumed run continues exactly
as an uninterrupted one. The shuffling comes from
:class:`ml.data.huncrc_tiles.EpochPermutation`, seeded the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW

from ml.huncrc_train import make_head


if TYPE_CHECKING:
    from ml.data.huncrc_tiles import TileBatch


class HunCRCHead(LightningModule):
    def __init__(
        self,
        kind: str,
        in_features: int = 1536,
        dropout: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.head = make_head(kind, in_features, dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self._loss_sum, self._loss_count = 0.0, 0

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x).squeeze(1)

    def on_train_epoch_start(self) -> None:
        torch.manual_seed(self.seed * 100_003 + self.current_epoch)
        self._loss_sum, self._loss_count = 0.0, 0

    def training_step(self, batch: TileBatch) -> Tensor:
        x, y, w = batch
        per_tile = self.criterion(self(x), y)
        if w is not None:
            per_tile = per_tile * w
        loss = per_tile.mean()
        self._loss_sum += float(loss.detach()) * len(y)
        self._loss_count += len(y)
        return loss

    @property
    def train_loss(self) -> float:
        """Mean training loss per tile over the epoch so far."""
        return self._loss_sum / self._loss_count

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
