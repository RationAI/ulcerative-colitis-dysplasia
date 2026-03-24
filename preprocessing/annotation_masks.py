import json
import tempfile
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from PIL import Image, ImageDraw
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


class JSONPolygonMask:
    """Parses JSON annotations and renders them into a multi-class mask array."""

    def __init__(
        self,
        json_path: Path,
        mask_size: tuple[int, int],
        scale_factor: float,
        target_groups: dict[str, int],
    ):
        self.json_path = json_path
        self.mask_size = mask_size  # (width, height)
        self.scale = scale_factor
        self.target_groups = target_groups

    def __call__(self) -> np.ndarray:
        mask_img = Image.new("L", self.mask_size, 0)
        draw = ImageDraw.Draw(mask_img)

        if not self.json_path.exists():
            return np.array(mask_img)

        with open(self.json_path) as f:
            data = json.load(f)

        for item in data.get("items", []):
            class_name = item.get("name")

            if class_name in self.target_groups:
                coords = item.get("coordinates", [])
                if not coords:
                    continue

                fill_value = self.target_groups[class_name]
                scaled_coords = [(c[0] * self.scale, c[1] * self.scale) for c in coords]
                draw.polygon(scaled_coords, fill=fill_value)

        return np.array(mask_img)


@ray.remote
def process_slide(
    slide_path: str,
    annot_path: Path,
    level: int,
    output_base: Path,
    target_groups: dict[str, int],
    logger: MLFlowLogger,
) -> str:
    slide_p = Path(slide_path)
    json_path = annot_path / f"{slide_p.stem}.json"

    if not json_path.exists():
        return f"Skipped: No JSON for {slide_p.name}"

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=level)
        full_dim = slide.level_dimensions[0]
        mask_dim = slide.level_dimensions[level]
        scale = mask_dim[0] / full_dim[0]

    parser = JSONPolygonMask(json_path, mask_dim, scale, target_groups)
    mask_array = parser()

    if mask_array.max() == 0:
        logger.log_stream(f"Empty mask for {slide_p.name}")

    vips_mask = pyvips.Image.new_from_array(mask_array)

    save_path = output_base / f"{slide_p.stem}.tiff"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    write_big_tiff(vips_mask, save_path, mpp_x=mpp_x, mpp_y=mpp_y)

    return f"Success: {slide_p.name}"


@with_cli_args(["+preprocessing=annotation_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset_path = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=config.dataset_uri)
    )
    annot_path = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=config.annot_mlflow_uri)
    )

    df = pd.read_csv(dataset_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        process_items(
            df["slide_path"].tolist(),
            process_item=process_slide, # type: ignore[arg-type]
            fn_kwargs={
                "annot_path": annot_path,
                "level": config.level,
                "output_base": tmpdir_path,
                "target_groups": config.target_groups,
                "logger": logger,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(str(tmpdir_path), "annot_masks")


if __name__ == "__main__":
    ray.init()
    main()
