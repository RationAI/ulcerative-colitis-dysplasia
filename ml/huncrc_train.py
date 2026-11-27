"""Heads, scaling, prediction and metrics for HunCRC heads on cached features.

The repo heads (``EmbeddingClassifier`` = dropout + Linear,
``MlpEmbeddingClassifier`` = dropout + 1536-512-256 + Linear) with the repo
defaults (``TrainConfig``: AdamW, lr 1e-4, batch 512, dropout 0.5). Features
stay memory-mapped and are standardised with a scaler fitted on the training
rows only. Training is ``ml.huncrc_head`` (Lightning).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from ml.modeling.decode_head.embedding_classifier import EmbeddingClassifier
from ml.modeling.decode_head.mlp_classifier import MlpEmbeddingClassifier


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
