import tempfile
from pathlib import Path
from typing import cast

import hydra
import mlflow
import pandas as pd
import pyvips
import ray
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
    slide_name = Path(slide_path).stem
    mask_file_name = f"{slide_name}.tiff"
    mask_path = output_path / mask_file_name

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level)

    image = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))
    mask = tissue_mask(image, mpp=(mpp_x + mpp_y) / 2)

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


@with_cli_args(["+preprocessing=tissue_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset_path = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=config.mlflow_uris.dataset)
    )
    dataset = pd.read_csv(dataset_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        process_items(
            dataset["slide_path"],
            process_item=process_slide,
            fn_kwargs={
                "level": config.level,
                "output_path": tmpdir_path,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(str(tmpdir_path), config.mlflow_artifact_path)


if __name__ == "__main__":
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):  # type: ignore[attr-defined, call-arg]
        main()
