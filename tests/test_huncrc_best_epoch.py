"""Best-epoch selection (ModelCheckpoint) agrees with np.argmax over float64 AUPRC.

The validation AUPRC per epoch is scripted, so the best epoch can be first,
in the middle, last, tied, or separated from another epoch only below float32
resolution. On the synthetic cohort of the reference test the best epoch is
always the last one, so that test alone cannot catch a selection bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from lightning import Callback, LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from ml.atomic_checkpoint_io import AtomicCheckpointIO
from ml.callbacks import YoudenThreshold, run_checkpoints
from ml.huncrc_head import HunCRCHead


if TYPE_CHECKING:
    from pathlib import Path


def _collate(
    b: list[tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, torch.Tensor, None]:
    return torch.stack([t[0] for t in b]), torch.stack([t[1] for t in b]), None


class TinyData(LightningDataModule):
    val_labels = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)

    def train_dataloader(self) -> DataLoader:
        g = torch.Generator().manual_seed(0)
        x, y = torch.randn(8, 4, generator=g), (torch.arange(8) % 2).float()
        return DataLoader(TensorDataset(x, y), batch_size=4, collate_fn=_collate)

    def val_dataloader(self) -> DataLoader:
        g = torch.Generator().manual_seed(1)
        x, y = torch.randn(6, 4, generator=g), torch.from_numpy(self.val_labels).float()
        return DataLoader(TensorDataset(x, y), batch_size=6, collate_fn=_collate)


class Scripted(HunCRCHead):
    def __init__(self, auprc: list[float]) -> None:
        super().__init__("linear", in_features=4)
        self.auprc = auprc

    def validation_metrics(self, scores: np.ndarray) -> tuple[float, float]:
        return 0.5, self.auprc[self.current_epoch]


class ValScores(Callback):
    """epoch_NN_val.npy, as the runner's EpochFiles writes it."""

    def __init__(self, d: Path, crash_at: int | None = None) -> None:
        self.d, self.crash_at = d, crash_at

    def on_train_epoch_start(self, trainer: Trainer, module: HunCRCHead) -> None:
        if trainer.current_epoch == self.crash_at:
            raise RuntimeError("simulated crash")

    def on_train_epoch_end(self, trainer: Trainer, module: HunCRCHead) -> None:
        np.save(
            self.d / f"epoch_{trainer.current_epoch:02d}_val.npy", module.val_scores
        )


def _fit(d: Path, auprc: list[float], crash_at: int | None = None) -> YoudenThreshold:
    ckpts = run_checkpoints(d)
    youden = YoudenThreshold(d, best=ckpts[0])
    last = d / "checkpoints" / "last.ckpt"
    trainer = Trainer(
        max_epochs=len(auprc), accelerator="cpu", logger=False,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, plugins=[AtomicCheckpointIO()],
        callbacks=[ValScores(d, crash_at), youden, *ckpts],
    )  # fmt: skip
    trainer.fit(Scripted(auprc), datamodule=TinyData(),
                ckpt_path=last if last.is_file() else None)  # fmt: skip
    return youden


F32_TIE = 0.98505521426285770  # equal to the next value in float32, not float64
CASES = {
    "middle": [0.80, 0.95, 0.90, 0.85],
    "first": [0.90, 0.80, 0.70],
    "last": [0.70, 0.80, 0.90],
    "exact tie keeps the earliest": [0.80, 0.95, 0.95, 0.90],
    "float32 tie, float64 larger later": [0.80, F32_TIE, F32_TIE + 3e-9, 0.90 - 0.2],
}


@pytest.mark.parametrize("name", list(CASES))
def test_best_epoch_is_argmax_of_float64(tmp_path: Path, name: str) -> None:
    auprc = CASES[name]
    if "float32" in name:
        assert np.float32(auprc[1]) == np.float32(auprc[2]) and auprc[2] > auprc[1]
    youden = _fit(tmp_path, auprc)
    assert youden.best_epoch == int(np.argmax(auprc))
    scores = np.load(tmp_path / f"epoch_{youden.best_epoch:02d}_val.npy")
    from ml.huncrc_eval import youden_threshold

    assert youden.threshold == youden_threshold(TinyData.val_labels, scores)


def test_best_epoch_survives_a_resume(tmp_path: Path) -> None:
    """Best in epoch 2, crash at the start of epoch 3, resume: still epoch 2."""
    auprc = [0.80, 0.95, 0.90, 0.85]
    with pytest.raises(RuntimeError, match="simulated crash"):
        _fit(tmp_path, auprc, crash_at=2)
    assert _fit(tmp_path, auprc).best_epoch == 1
