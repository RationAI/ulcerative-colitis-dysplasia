"""Run files written so that an interrupted run never leaves a half-written one."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch


if TYPE_CHECKING:
    from pathlib import Path


def save(obj: Any, path: Path, kind: str = "json") -> None:
    """Atomic write (tmp + rename), so an interrupted run never leaves half files."""
    tmp = path.with_name(path.name + ".tmp")
    if kind == "npy":
        with open(tmp, "wb") as f:
            np.save(f, obj)
    elif kind == "pt":
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)
