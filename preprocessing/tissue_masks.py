import ray
import pandas as pd
import pyvips
from pathlib import Path
from typing import cast

import hydra
from omegaconf import DictConfig
from openslide import OpenSlide

from rationai.masks import (
    process_items,
    slide_resolution,
    tissue_mask,
    write_big_tiff,
)
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(memory=4 * 1024**3)
def process_slide(slide_path: str, level: int, output_path: Path) -> None:
    """Generates a tissue mask from a local slide and saves it locally."""
    slide_name = Path(slide_path).stem
    mask_file_name = f"{slide_name}.tiff"
    mask_path = output_path / mask_file_name

    # Check if slide actually exists locally before opening
    if not Path(slide_path).exists():
        print(f"❌ Slide not found: {slide_path}")
        return

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level)

    image = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))
    mask = tissue_mask(image, mpp=(mpp_x + mpp_y) / 2)

    write_big_tiff(mask, path=str(mask_path), mpp_x=mpp_x, mpp_y=mpp_y)


@with_cli_args([])
@hydra.main(config_path="conf", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    csv_input_path = Path(config.dataset.local_path)
    dataset = pd.read_csv(csv_input_path)

    local_mask_dir = Path(config.tissue_mask.output_dir)
    local_mask_dir.mkdir(parents=True, exist_ok=True)

    process_items(
        dataset["slide_path"],
        process_item=process_slide,
        fn_kwargs={
            "level": config.tissue_mask.level,
            "output_path": local_mask_dir,
        },
        max_concurrent=config.tissue_mask.max_concurrent,
    )

    # logger.log_artifacts(local_dir=str(local_mask_dir), artifact_path="tissue_masks")


if __name__ == "__main__":
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
