from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset


class LabelMode(Enum):
    MIXED = "mixed"
    HIGH = "high"
    LOW = "low"
    HIGH_LOW = "high_low"


@dataclass(frozen=True)
class LabelColumns:
    """Names of the tile columns holding the dysplasia annotation coverage.

    The IKEM tiling names them ``LG Dysplasia`` / ``HG Dysplasia`` (the
    defaults); the HunCRC tiling names them ``lowgrade_dysplasia`` /
    ``highgrade_dysplasia``, because the tiling takes the column names
    verbatim from the annotation files. ``annotation`` is the combined
    coverage column used to tell an annotated tile from an unannotated one.
    """

    low: str = "LG Dysplasia"
    high: str = "HG Dysplasia"
    annotation: str = "annotation"


DEFAULT_LABEL_COLUMNS = LabelColumns()


def as_label_columns(
    columns: "LabelColumns | Mapping[str, str] | None",
) -> LabelColumns:
    """Coerce a config value into :class:`LabelColumns`.

    Hydra may hand over a ``DictConfig`` rather than an instantiated
    dataclass, so a plain mapping is accepted too.
    """
    if columns is None:
        return DEFAULT_LABEL_COLUMNS
    if isinstance(columns, LabelColumns):
        return columns
    return LabelColumns(**dict(columns))


def process_slides(
    slides: HFDataset,
    val_fold: int | None = None,
    is_val: bool = False,
) -> HFDataset:
    if val_fold is not None:
        if is_val:
            slides = slides.filter(lambda x: x["fold"] == val_fold)
        else:
            slides = slides.filter(lambda x: x["fold"] != val_fold)

    slides = slides.map(lambda x: {"name": Path(x["path"]).stem})
    return slides


def get_label(
    tile: dict[str, Any],
    mode: LabelMode,
    columns: LabelColumns = DEFAULT_LABEL_COLUMNS,
) -> torch.Tensor:
    high = tile[columns.high] > 0.5
    low = tile[columns.low] > 0.5

    match mode:
        case LabelMode.MIXED:
            return torch.tensor(high or low).float()
        case LabelMode.HIGH:
            return torch.tensor(high).float()
        case LabelMode.LOW:
            return torch.tensor(low).float()
        case LabelMode.HIGH_LOW:
            return torch.tensor([high, low]).float()
