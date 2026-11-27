"""One CV fold of a HunCRC feature cache as a LightningDataModule.

``setup`` picks the fold's training and validation rows (negatives capped per
slide in training only), fits the scaler on the training rows or reloads it,
and computes the class weights. A resumed run must find the rows it saved
before, otherwise it stops. Training batches follow the reference loop's
shuffling; validation goes through the rows in sorted order, in blocks of
``chunk`` rows, as the reference ``predict`` did.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import BatchSampler, DataLoader

from ml.data.huncrc_folds import fold_rows
from ml.data.huncrc_tiles import TileRows, as_batch, train_loader
from ml.huncrc_heads import Scaler, balanced_weights
from ml.run_files import save


if TYPE_CHECKING:
    from ml.data.huncrc_features import FeatureSet
    from ml.data.huncrc_tiles import TileBatch


class HunCRCFoldData(LightningDataModule):
    def __init__(
        self,
        features: FeatureSet,
        fold: int,
        run_dir: Path | str,
        safe_slides: set[str],
        class_weighting: str | None = None,
        neg_cap: int | None = 5000,
        cap_seed: int = 0,
        smoke_tiles_per_slide: int | None = None,
        batch_size: int = 512,
        chunk: int = 65536,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.fs, self.fold, self.run_dir = features, fold, Path(run_dir)
        self.safe, self.class_weighting = safe_slides, class_weighting
        self.neg_cap, self.cap_seed = neg_cap, cap_seed
        self.smoke = smoke_tiles_per_slide
        self.batch_size, self.chunk, self.seed = batch_size, chunk, seed
        self.scaler: Scaler | None = None

    def setup(self, stage: str) -> None:
        if self.scaler is not None:  # called again by the Trainer
            return
        d = self.run_dir
        d.mkdir(parents=True, exist_ok=True)
        tr, val, cnt = fold_rows(
            self.fs, self.fold, self.smoke, self.neg_cap, self.cap_seed, self.safe
        )
        if (d / "train_rows.npy").is_file():  # resume: rows must be what we trained on
            assert np.array_equal(np.load(d / "train_rows.npy"), tr), (
                "train rows changed; delete the run dir"
            )
            assert np.array_equal(np.load(d / "val_rows.npy"), val), (
                "val rows changed; delete the run dir"
            )
        else:
            save(tr, d / "train_rows.npy", "npy")
            save(val, d / "val_rows.npy", "npy")
            save(
                {
                    **cnt,
                    "neg_cap": self.neg_cap,
                    "cap_seed": self.cap_seed,
                    "smoke_tiles_per_slide": self.smoke,
                },
                d / "counts.json",
            )
        x, y = self.fs.features, self.fs.labels
        if (d / "scaler.npz").is_file():
            z = np.load(d / "scaler.npz")
            sc = Scaler(z["mean"], z["std"])
        else:
            sc = Scaler.fit(x, tr)
            np.savez(d / "scaler.npz", mean=sc.mean, std=sc.std)
        # blocks of sorted rows, exactly as the reference predict() scored them
        assert np.all(np.diff(val) > 0), "validation rows must be sorted"
        self.train_rows, self.val_rows, self.counts = tr, val, cnt
        self.class_weights = (
            balanced_weights(y[tr]) if self.class_weighting == "balanced" else None
        )
        self.scaler = sc

    @property
    def val_labels(self) -> np.ndarray:
        return self.fs.labels[self.val_rows]

    def train_dataloader(self) -> DataLoader[TileBatch]:
        assert self.scaler is not None
        return train_loader(
            self.fs.features,
            self.fs.labels,
            self.train_rows,
            self.scaler,
            self.batch_size,
            self.seed,
            self.class_weights,
        )

    def val_dataloader(self) -> DataLoader[TileBatch]:
        assert self.scaler is not None
        return DataLoader(
            TileRows(self.fs.features, self.fs.labels, self.scaler),
            batch_sampler=BatchSampler(self.val_rows, self.chunk, drop_last=False),
            collate_fn=as_batch,
            generator=torch.Generator(),
        )
