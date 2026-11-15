"""Single inference engine shared by every "run the model" entrypoint.

One forward pass over a chosen tile source (a val fold, a labeled split, or
an unlabeled held-out set) produces one predictions parquet with a row per
tile: ``slide_name``, ``x``, ``y``, one probability column per class, and
``hg_label`` when the source carries ground truth. Downstream steps are cheap
CPU consumers of that parquet: ``f1_scan`` (threshold selection / Step 2, plus
held-out F1 / Step 3) and ``save_heatmaps`` (rasterize per slide).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader

from ml.data.datasets.labels import LabelMode
from ml.heatmap import save_heatmaps
from ml.predict_output import _class_names


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.nn import Module

    from ml.typing import Input, PredictInput


def _prob(probs: Tensor, index: int) -> Tensor:
    # A single-class head yields (B,); a high/low head yields (B, 2). The i-th
    # class is ``probs[:, i]`` for a 2D head, ``probs`` itself for a 1D head.
    return probs[:, index] if probs.ndim == 2 else probs


def collect(
    loader: DataLoader[Input] | DataLoader[PredictInput],
    model: Module,
    device: torch.device,
    include_labels: bool,
    mode: LabelMode | str,
) -> list[dict[str, object]]:
    """Forward the head over ``loader`` and collect one dict row per tile.

    ``include_labels`` selects the labeled batch layout (embedding, label,
    metadata) and emits an ``hg_label`` column. The HG channel is always
    index 0 of the label / head output, so this works for both ``high`` and
    ``high_low`` models.
    """
    model.to(device).eval()
    class_names = _class_names(mode)

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch[0].to(device)
            metadata = batch[-1]
            probs = torch.sigmoid(model(embeddings)).cpu()

            if include_labels:
                label = batch[1]
                hg_label = (label[:, 0] if label.ndim == 2 else label).cpu()

            slide_names = metadata["slide_name"]
            xs = [int(v) for v in metadata["x"].tolist()]
            ys = [int(v) for v in metadata["y"].tolist()]
            for i in range(probs.shape[0]):
                row: dict[str, object] = {
                    "slide_name": slide_names[i],
                    "x": xs[i],
                    "y": ys[i],
                }
                for col, name in enumerate(class_names):
                    row[name] = float(_prob(probs, col)[i])
                if include_labels:
                    row["hg_label"] = int(hg_label[i])
                rows.append(row)
    return rows


def write_parquet(rows: list[dict[str, object]], output_path: str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(out)
    return out


def f1_at(
    predictions_path: str | Path,
    threshold: float,
    prob_column: str = "prob_high",
    label_column: str = "hg_label",
) -> dict[str, float]:
    """Precision/recall/F1 of ``prob_column >= threshold`` at a fixed cut."""
    df = pd.read_parquet(predictions_path)
    prob = torch.tensor(df[prob_column].to_numpy(dtype=float))
    label = torch.tensor(df[label_column].to_numpy(dtype=int))
    preds = (prob >= threshold).int()
    tp = int(((preds == 1) & (label == 1)).sum())
    fp = int(((preds == 1) & (label == 0)).sum())
    fn = int(((preds == 0) & (label == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def f1_scan(
    predictions_path: str | Path,
    prob_column: str = "prob_high",
    label_column: str = "hg_label",
    num_thresholds: int = 500,
) -> dict[str, float]:
    """Sweep thresholds over ``prob_column`` and return argmax-F1 statistics.

    Reads the predictions parquet (so F1 needs no GPU). F1 is tile-level
    against ``label_column``; the sweep returns the threshold, precision,
    recall and F1 at the argmax point plus per-threshold curve metadata.
    """
    df = pd.read_parquet(predictions_path)
    prob = torch.tensor(df[prob_column].to_numpy(dtype=float))
    label = torch.tensor(df[label_column].to_numpy(dtype=int))

    if prob.numel() == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    grid = torch.linspace(prob.min(), prob.max(), num_thresholds)
    best: dict[str, float] = {}
    for thr in grid:
        preds = (prob >= thr).int()
        tp = int(((preds == 1) & (label == 1)).sum())
        fp = int(((preds == 1) & (label == 0)).sum())
        fn = int(((preds == 0) & (label == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        if not best or f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return best


def heatmap(
    predictions_path: str | Path,
    slides: Iterable[dict[str, object]],
    value_columns: list[str],
    save_dir: str | Path,
) -> list[Path]:
    """Rasterize the predictions parquet into per-slide heatmaps."""
    return save_heatmaps(predictions_path, slides, value_columns, save_dir)


__all__ = [
    "LabelMode",
    "_class_names",
    "_prob",
    "collect",
    "f1_scan",
    "heatmap",
    "write_parquet",
]
