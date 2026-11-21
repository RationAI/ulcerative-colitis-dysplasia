"""Minimal training loop for heads on cached HunCRC features (CPU friendly).

Uses the repo heads (``EmbeddingClassifier`` = dropout + Linear,
``MlpEmbeddingClassifier`` = dropout + 1536-512-256 + Linear) with the repo
defaults: AdamW, lr 1e-4, batch 512, dropout 0.5, BCEWithLogits.

Features stay memory-mapped. Each epoch draws a random permutation, reads it
in sorted "super-batches" of random rows (sequential-ish IO), standardises
them with a scaler fitted on the training rows only, and splits them into
minibatches in permuted order.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from ml.modeling.decode_head.embedding_classifier import EmbeddingClassifier
from ml.modeling.decode_head.mlp_classifier import MlpEmbeddingClassifier


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 512
    epochs: int = 1
    dropout: float = 0.5
    weight_decay: float = 0.01  # AdamW default
    seed: int = 0
    super_batch: int = 65536


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, rows: np.ndarray, chunk: int = 65536) -> Scaler:
        """Streaming mean/std (float64 accumulators) over ``x[rows]``."""
        rows = np.sort(rows)
        d = x.shape[1]
        s = np.zeros(d)
        ss = np.zeros(d)
        for i in range(0, len(rows), chunk):
            b = np.asarray(x[rows[i : i + chunk]], dtype=np.float64)
            s += b.sum(0)
            ss += (b * b).sum(0)
        n = len(rows)
        mean = s / n
        var = np.maximum(ss / n - mean * mean, 0.0)
        std = np.sqrt(var)
        std[std < 1e-8] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, b: np.ndarray) -> np.ndarray:
        return (b - self.mean) / self.std


def make_head(kind: str, in_features: int = 1536, dropout: float = 0.5) -> nn.Module:
    if kind == "linear":
        return EmbeddingClassifier(
            in_features=in_features, out_features=1, dropout=dropout
        )
    if kind == "mlp":
        return MlpEmbeddingClassifier(
            in_features=in_features,
            out_features=1,
            hidden_dims=(512, 256),
            dropout=dropout,
        )
    raise ValueError(kind)


@dataclass
class History:
    epoch_loss: list[float] = field(default_factory=list)
    seconds: float = 0.0


def train_head(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    scaler: Scaler,
    config: TrainConfig,
    on_epoch: Callable[[int, nn.Module], None] | None = None,
) -> tuple[nn.Module, History]:
    """Train on ``x[rows]`` with targets ``y[rows]`` (``y`` indexed like ``x``)."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = make_head(kind, x.shape[1], config.dropout)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    hist = History()
    t0 = time.time()
    for epoch in range(config.epochs):
        model.train()
        perm = rows[rng.permutation(len(rows))]
        total, count = 0.0, 0
        for s in range(0, len(perm), config.super_batch):
            chunk = perm[s : s + config.super_batch]
            order = np.argsort(chunk)
            xb_all = np.empty((len(chunk), x.shape[1]), dtype=np.float32)
            xb_all[order] = scaler(np.asarray(x[chunk[order]]))
            yb_all = y[chunk].astype(np.float32)
            for b in range(0, len(chunk), config.batch_size):
                xb = torch.from_numpy(xb_all[b : b + config.batch_size])
                yb = torch.from_numpy(yb_all[b : b + config.batch_size])
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb).squeeze(1), yb)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(yb)
                count += len(yb)
        hist.epoch_loss.append(total / count)
        if on_epoch is not None:
            on_epoch(epoch, model)
    hist.seconds = time.time() - t0
    return model, hist


@torch.no_grad()
def predict(
    model: nn.Module,
    x: np.ndarray,
    rows: np.ndarray,
    scaler: Scaler,
    chunk: int = 65536,
) -> np.ndarray:
    """Sigmoid scores for ``x[rows]`` (same order as ``rows``), eval mode."""
    model.eval()
    out = np.empty(len(rows), dtype=np.float32)
    order = np.argsort(rows)
    srt = rows[order]
    for s in range(0, len(srt), chunk):
        b = torch.from_numpy(scaler(np.asarray(x[srt[s : s + chunk]])))
        out[order[s : s + chunk]] = torch.sigmoid(model(b).squeeze(1)).numpy()
    return out


def binary_metrics(
    y: np.ndarray, score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = score >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "auroc": float(roc_auc_score(y, score)) if 0 < y.mean() < 1 else float("nan"),
        "auprc": float(average_precision_score(y, score)),
        "accuracy": (tp + tn) / len(y),
        "precision": prec,
        "recall": rec,
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": 2 * prec * rec / (prec + rec) if tp else 0.0,
        "prevalence": float(y.mean()),
        "n": len(y),
    }


def balanced_weights(y: np.ndarray) -> tuple[float, float]:
    """(w_neg, w_pos) so both classes carry equal total weight and the mean weight is 1."""
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    return n / (2 * n_neg), n / (2 * n_pos)


def fit_epochs(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    scaler: Scaler,
    config: TrainConfig,
    class_weights: tuple[float, float] | None = None,
    resume: dict | None = None,
):
    """Yield ``(epoch, model, state, train_loss)`` after every epoch.

    Resumable: each epoch's shuffling and dropout are seeded from
    ``(seed, epoch)`` only, so restarting from a saved ``state`` (model +
    optimizer, as yielded) continues exactly as an uninterrupted run would.
    ``class_weights = (w_neg, w_pos)`` weights the per-tile BCE.
    """
    torch.manual_seed(config.seed)
    model = make_head(kind, x.shape[1], config.dropout)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    start = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        start = resume["epoch"] + 1
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    cw = None if class_weights is None else np.asarray(class_weights, dtype=np.float32)
    for epoch in range(start, config.epochs):
        rng = np.random.default_rng([config.seed, epoch])
        torch.manual_seed(config.seed * 100_003 + epoch)
        model.train()
        perm = rows[rng.permutation(len(rows))]
        total, count = 0.0, 0
        for s in range(0, len(perm), config.super_batch):
            chunk = perm[s : s + config.super_batch]
            order = np.argsort(chunk)
            xb_all = np.empty((len(chunk), x.shape[1]), dtype=np.float32)
            xb_all[order] = scaler(np.asarray(x[chunk[order]]))
            yb_all = y[chunk].astype(np.float32)
            wb_all = None if cw is None else cw[y[chunk].astype(np.int64)]
            for b in range(0, len(chunk), config.batch_size):
                xb = torch.from_numpy(xb_all[b : b + config.batch_size])
                yb = torch.from_numpy(yb_all[b : b + config.batch_size])
                opt.zero_grad(set_to_none=True)
                per_tile = loss_fn(model(xb).squeeze(1), yb)
                if wb_all is not None:
                    per_tile = per_tile * torch.from_numpy(
                        wb_all[b : b + config.batch_size]
                    )
                loss = per_tile.mean()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(yb)
                count += len(yb)
        #: deep copy: state_dict() returns live references that later epochs mutate
        state = copy.deepcopy(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
            }
        )
        yield epoch, model, state, total / count
