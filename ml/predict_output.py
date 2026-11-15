from pathlib import Path

from datasets import Dataset
from torch import Tensor

from ml.data.datasets.labels import LabelMode
from ml.typing import MetadataBatch


def _class_names(mode: LabelMode | str) -> list[str]:
    match LabelMode(mode):
        case LabelMode.HIGH_LOW:
            return ["prob_high", "prob_low"]
        case LabelMode.HIGH:
            return ["prob_high"]
        case LabelMode.LOW:
            return ["prob_low"]
        case LabelMode.MIXED:
            return ["prob_positive"]


def save_predictions(
    predictions: list[Tensor],
    metadata: list[MetadataBatch],
    output_path: str,
    mode: LabelMode | str,
) -> None:
    """Flatten (predictions, metadata) batches into a per-tile parquet.

    Each row is one tile: ``slide_name``, ``x``, ``y`` plus one probability
    column per class (order matches ``LabelMode``). Predictions and metadata
    are matched by batch position.
    """
    class_names = _class_names(mode)
    rows: list[dict[str, object]] = []
    for batch_predictions, batch_metadata in zip(predictions, metadata, strict=True):
        probs = batch_predictions.detach().cpu().numpy()
        if probs.ndim == 1:
            probs = probs[:, None]
        slide_names = batch_metadata["slide_name"]
        xs = [int(v) for v in batch_metadata["x"].tolist()]
        ys = [int(v) for v in batch_metadata["y"].tolist()]
        for i in range(probs.shape[0]):
            row: dict[str, object] = {
                "slide_name": slide_names[i],
                "x": xs[i],
                "y": ys[i],
            }
            for col, name in enumerate(class_names):
                row[name] = float(probs[i, col])
            rows.append(row)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(out)
