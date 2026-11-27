"""Tile rows of a memory-mapped HunCRC feature cache, batched as the reference loop did.

:class:`EpochPermutation` draws a permutation of the rows from ``(seed, epoch)``;
a ``BatchSampler`` over it cuts it into consecutive batches. :class:`TileRows` reads a batch in sorted row order
(sequential-ish IO on the memmap), standardises it and returns it in batch order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler


if TYPE_CHECKING:
    from collections.abc import Iterator

    from ml.huncrc_train import Scaler

TileBatch = tuple[Tensor, Tensor, Tensor | None]


class TileRows(Dataset[TileBatch]):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        scaler: Scaler,
        class_weights: tuple[float, float] | None = None,
    ) -> None:
        self.x, self.y, self.scaler = x, y, scaler
        self.cw = (
            None
            if class_weights is None
            else np.asarray(class_weights, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> TileBatch:
        return self.__getitems__([idx])

    def __getitems__(self, rows: list[int] | np.ndarray) -> TileBatch:
        rows = np.asarray(rows)
        order = np.argsort(rows)
        xb = np.empty((len(rows), self.x.shape[1]), dtype=np.float32)
        xb[order] = self.scaler(np.asarray(self.x[rows[order]]))
        labels = self.y[rows]
        wb = (
            None
            if self.cw is None
            else torch.from_numpy(self.cw[labels.astype(np.int64)])
        )
        return torch.from_numpy(xb), torch.from_numpy(labels.astype(np.float32)), wb


def as_batch(batch: TileBatch) -> TileBatch:
    """collate_fn: :meth:`TileRows.__getitems__` already returns a whole batch."""
    return batch


class EpochPermutation(Sampler[int]):
    """The rows in a permutation drawn from ``(seed, epoch)``.

    Lightning calls ``set_epoch`` on ``dataloader.sampler`` and on
    ``batch_sampler.sampler`` before every epoch (also after a resume), so wrap
    this in a ``torch.utils.data.BatchSampler``.
    """

    def __init__(self, rows: np.ndarray, seed: int) -> None:
        self.rows, self.seed = rows, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng([self.seed, self.epoch])
        return iter(self.rows[rng.permutation(len(self.rows))].tolist())
