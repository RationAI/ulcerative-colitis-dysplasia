"""LightningModule for the HunCRC tile heads (linear or MLP on frozen embeddings).

Training matches the hand-written loop of 8b5955c bit for bit: AdamW over the
head's parameters, BCE with logits per tile (optionally class-weighted), and
dropout seeded from ``(seed, epoch)`` only, so a resumed run continues exactly
as an uninterrupted one. The shuffling comes from
:class:`ml.data.huncrc_tiles.EpochPermutation`, seeded the same way.

torchmetrics measures the training batches for monitoring only
(:meth:`HunCRCHead.pop_monitoring`). It returns float32, so it never
writes a run file or selects an epoch; those use ml.huncrc_eval in float64.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from ml.huncrc_eval import weighted_auroc_ap
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
        #: monitoring only (float32): never in a run file, never selects an epoch
        self.train_monitor = MetricCollection(
            {"auroc": BinaryAUROC(), "auprc": BinaryAveragePrecision()},
            prefix="monitor_train_",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x).squeeze(1)

    def on_train_epoch_start(self) -> None:
        torch.manual_seed(self.seed * 100_003 + self.current_epoch)
        self._loss_sum, self._loss_count = 0.0, 0

    def training_step(self, batch: TileBatch) -> Tensor:
        x, y, w = batch
        logits = self(x)
        per_tile = self.criterion(logits, y)
        if w is not None:
            per_tile = per_tile * w
        loss = per_tile.mean()
        self._loss_sum += float(loss.detach()) * len(y)
        self._loss_count += len(y)
        self.train_monitor.update(torch.sigmoid(logits.detach()), y.int())
        return loss

    def pop_monitoring(self) -> dict[str, float]:
        """The epoch's monitoring metrics (torchmetrics, float32), then reset."""
        out = {k: float(v) for k, v in self.train_monitor.compute().items()}
        self.train_monitor.reset()
        return out

    def on_validation_epoch_start(self) -> None:
        self._val_scores: list[Tensor] = []

    def validation_step(self, batch: TileBatch) -> None:
        self._val_scores.append(torch.sigmoid(self(batch[0])))

    def on_validation_epoch_end(self) -> None:
        #: sigmoid scores of the validation rows, in the order of the loader
        self.val_scores = torch.cat(self._val_scores).numpy()
        self.val_auroc, self.val_auprc = self.validation_metrics(self.val_scores)
        # float64, so ModelCheckpoint tells apart values equal in float32
        self.log("validation/auroc", torch.tensor(self.val_auroc, dtype=torch.float64))
        self.log("validation/auprc", torch.tensor(self.val_auprc, dtype=torch.float64))

    def validation_metrics(self, scores: np.ndarray) -> tuple[float, float]:
        """Validation AUROC and AUPRC in float64 (ml.huncrc_eval)."""
        return weighted_auroc_ap(self.trainer.datamodule.val_labels, scores)

    @property
    def train_loss(self) -> float:
        """Mean training loss per tile over the epoch so far."""
        return self._loss_sum / self._loss_count

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
