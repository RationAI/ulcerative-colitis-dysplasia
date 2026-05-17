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


def get_label(slide_metadata: dict[str, Any], mode: LabelMode) -> torch.Tensor:
    match mode:
        case LabelMode.MIXED:
            return torch.tensor(
                slide_metadata["HG Dysplasia"] > 0 or slide_metadata["LG Dysplasia"] > 0
            ).float()
        case LabelMode.HIGH:
            return torch.tensor(slide_metadata["HG Dysplasia"] > 0).float()
        case LabelMode.LOW:
            return torch.tensor(slide_metadata["LG Dysplasia"] > 0).float()
        case LabelMode.HIGH_LOW:
            hg = slide_metadata["HG Dysplasia"]
            lg = slide_metadata["LG Dysplasia"]

            if hg == 0 and lg == 0:
                return torch.tensor(0).long()
            elif lg > hg:
                return torch.tensor(1).long()
            else:
                return torch.tensor(2).long()
