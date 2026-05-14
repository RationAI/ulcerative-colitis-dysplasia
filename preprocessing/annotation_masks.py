import re
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from rasterio.features import rasterize
from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.parsers import EMPAIAParser
from shapely.affinity import scale as shapely_scale


def create_annotation_mask(
    annot_path: Path,
    mask_dim: tuple[int, int],
    scale: float,
    target_groups: dict[str, int],
) -> np.ndarray:
    """Create a multi-class annotation mask from an EMPAIA JSON annotation file.

    Args:
        annot_path: Path to the EMPAIA JSON annotation file.
        mask_dim: Dimensions of the output mask as (width, height).
        scale: Scale factor to apply to annotation coordinates.
        target_groups: Mapping of annotation label names to mask integer values.

    Returns:
        Numpy array of shape (height, width) with dtype uint8.
    """
    parser = EMPAIAParser(annot_path)
    width, height = mask_dim
    shapes: list[tuple] = []

    for label, value in target_groups.items():
        for poly in parser.get_polygons(name=rf"^{re.escape(label)}$"):
            if poly.is_empty or poly.area == 0:
                continue
            scaled = shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
            if not scaled.is_valid:
                scaled = scaled.buffer(0)
            if not scaled.is_empty:
                shapes.append((scaled, value))

    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    return rasterize(
        shapes,
        out_shape=(height, width),
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )


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

    mask_array = create_annotation_mask(json_path, mask_dim, scale, target_groups)

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
    annot_path = Path(config.annot_path)
    output_base = Path(config.output_dir)

    df = pd.read_csv(dataset_path)

    process_items(
        df["slide_path"].tolist(),
        process_item=process_slide,  # type: ignore[arg-type]
        fn_kwargs={
            "annot_path": annot_path,
            "level": config.level,
            "output_base": output_base,
            "target_groups": config.target_groups,
            "logger": logger,
        },
        max_concurrent=config.max_concurrent,
    )

    logger.log_artifacts(str(output_base), "annot_masks")


if __name__ == "__main__":
    ray.init(runtime_env={"excludes": [".git", ".venv"]})
    try:
        main()
    finally:
        ray.shutdown()
