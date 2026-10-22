"""Entry point for the dysplasia U-Net experiment.

    uv run python -m ml.unet.train                  # all defaults
    uv run python -m ml.unet.train ++trainer.max_epochs=20 ++data.batch_size=8

Every parameter has a default in configs/unet.yaml (this file only wires
config -> DataModule -> model -> Trainer, the same shape as ml/__main__.py).
Independent of the embedding pipeline: own config name, own output dir; shared
read-only: configs/base.yaml, ml/data.DataModule, rationai.mlkit Trainer.

After fit, writes {output_dir}/demo_{split}.png -- a few (input / target /
prediction) side-by-side patches to eyeball without opening MLflow.
"""

from __future__ import annotations

from pathlib import Path
from random import randint

import hydra
import matplotlib
import numpy as np
import torch
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

from ml.data import DataModule


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


def demo_grid(net, val_ds, out_path: Path, n: int = 4) -> None:
    """Save n (input | target | prediction) rows to one PNG."""
    import matplotlib.pyplot as plt

    net.eval()
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.5 * n))
    with torch.no_grad():
        for i in range(n):
            img, target, _valid = val_ds[i]
            pred = (torch.sigmoid(net(img[None])) > 0.5).float()
            for j, (arr, title) in enumerate(
                ((img, "input"), (target, "target"), (pred, "pred"))
            ):
                ax = axes[i][j]
                ax.imshow(arr.squeeze(0).numpy())
                ax.set_title(f"{i + 1}: {title}", fontsize=8)
                ax.axis("off")
    fig.suptitle("dysplasia U-Net -- val demo (input | target | prediction)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"demo grid -> {out_path}")


@hydra.main(config_path="../../configs", config_name="unet", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    seed_everything(config.seed, workers=True)

    data = hydra.utils.instantiate(
        config.data,
        _recursive_=False,  # instantiate only the DataModule, not the datasets
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    trainer.fit(model, datamodule=data, ckpt_path=config.checkpoint)

    out_dir = Path(config.output_dir)
    demo_grid(model.net, data.val, out_dir / "demo_val.png")


if __name__ == "__main__":
    main()
