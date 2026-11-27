"""Operating threshold of a fold: Youden's J on the best epoch's validation scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from lightning import Callback

from ml.huncrc_eval import youden_threshold


if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


class YoudenThreshold(Callback):
    """After training: the best epoch (max validation AUPRC) and its threshold.

    Reads the per-epoch run files (``epoch_NN.json``, ``epoch_NN_val.npy``) of
    ``run_dir``, so a resumed run sees the epochs of the earlier attempt too.
    The threshold maximises sensitivity + specificity - 1 on the validation
    fold; ``best_epoch`` is 0-based.
    """

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.best_epoch: int | None = None
        self.threshold: float | None = None

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        d = self.run_dir
        hist = [
            json.loads((d / f"epoch_{e:02d}.json").read_text())
            for e in range(trainer.max_epochs)
        ]
        self.best_epoch = int(np.argmax([h["val_auprc"] for h in hist]))
        self.threshold = youden_threshold(
            trainer.datamodule.val_labels,
            np.load(d / f"epoch_{self.best_epoch:02d}_val.npy"),
        )
