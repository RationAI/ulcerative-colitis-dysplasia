"""Checkpoint writes that are atomic on network filesystems.

Lightning's ``TorchCheckpointIO`` writes through an fsspec transaction whose
temp file lives in ``/tmp``; the final ``shutil.move`` onto /home (NFS) or
/mnt/projects (CephFS) is then a copy, which a crash can leave truncated. Here
the temp file sits next to the target and ``os.replace`` renames it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO


class AtomicCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: Any, storage_options: Any = None
    ) -> None:
        if storage_options is not None:
            raise TypeError("AtomicCheckpointIO takes no storage_options")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
