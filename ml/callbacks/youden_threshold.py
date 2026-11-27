"""Operating threshold of a fold: Youden's J on the best epoch's validation scores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from lightning import Callback

from ml.huncrc_eval import youden_threshold


if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint


class YoudenThreshold(Callback):
    """After training: the best epoch and its threshold.

    The best epoch is the one ``best`` (a monitoring ModelCheckpoint, see
    :func:`ml.callbacks.run_checkpoints.run_checkpoints`) kept; its state is
    restored on resume, so earlier attempts count too. The threshold maximises
    sensitivity + specificity - 1 on that epoch's validation scores
    (``epoch_NN_val.npy`` in ``run_dir``). ``best_epoch`` is 0-based.
    """

    def __init__(self, run_dir: Path | str, best: ModelCheckpoint) -> None:
        self.run_dir = Path(run_dir)
        self.best = best
        self.best_epoch: int | None = None
        self.threshold: float | None = None

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # saved at the end of its epoch, so "epoch" is that epoch's 0-based index
        ckpt = torch.load(self.best.best_model_path, weights_only=False)
        self.best_epoch = int(ckpt["epoch"])
        self.threshold = youden_threshold(
            trainer.datamodule.val_labels,
            np.load(self.run_dir / f"epoch_{self.best_epoch:02d}_val.npy"),
        )
