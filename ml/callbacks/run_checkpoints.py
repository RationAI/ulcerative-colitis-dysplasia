"""The two checkpoints of a fold: the best epoch and the resume point."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint


MONITOR = "validation/auprc"  # float64 (ml.huncrc_eval), logged by HunCRCHead


def run_checkpoints(run_dir: Path | str) -> list[ModelCheckpoint]:
    """``[best, last]`` under ``<run_dir>/checkpoints``, both saved at train-epoch end.

    best: the epoch with the highest validation AUPRC; a strictly greater
    value is needed to replace it, so ties keep the earliest epoch (as
    ``np.argmax``). The metric stays float64, so values equal in float32 are
    still told apart. last: overwritten after every epoch, the resume point.
    Not ``save_last``: in Lightning 2.6 that refreshes last.ckpt only when a
    top-k checkpoint is saved. Order matters: best comes first, so last.ckpt
    holds the best-checkpoint state of the same epoch. Both save after the
    other callbacks' run files (Lightning runs checkpoint callbacks last); by
    default they would save at validation end, before them.
    """
    d = Path(run_dir) / "checkpoints"
    best = ModelCheckpoint(
        dirpath=d,
        filename="best-{epoch:02d}",
        monitor=MONITOR,
        mode="max",
        save_top_k=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )
    last = ModelCheckpoint(
        dirpath=d,
        filename="last",
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    return [best, last]
