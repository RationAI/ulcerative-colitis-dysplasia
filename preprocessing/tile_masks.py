"""Script for creating outlines of the tiles and also masks based on tiling percentages."""

from pathlib import Path

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks import process_items, tile_mask
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder
from ratiopath.openslide import OpenSlide


@ray.remote
def process_slide(
    slide: pd.Series,
    percentage_cols: list[str],
    output_path: Path,
    tiles: pd.DataFrame,
    level: int,
) -> None:
    slide_tiles = tiles[tiles["slide_id"] == slide["id"]].copy()
    with OpenSlide(slide["path"]) as slide_wsi:
        mask_extent_x, mask_extent_y = slide_wsi.level_dimensions[level]
        mpp_x, mpp_y = slide_wsi.slide_resolution(level)

    tile_extent_x = int(slide["tile_extent_x"] / mpp_x * slide["mpp_x"])
    tile_extent_y = int(slide["tile_extent_y"] / mpp_y * slide["mpp_y"])
    stride_x = int(slide["stride_x"] / mpp_x * slide["mpp_x"])
    stride_y = int(slide["stride_y"] / mpp_y * slide["mpp_y"])

    # Convert to target-level pixels using the MPP ratio, then snap to the target
    # stride grid so the outlines and MaskBuilder use the same tile positions.
    slide_tiles["x"] = (
        slide_tiles["x"] * slide["mpp_x"] / mpp_x / stride_x
    ).round().astype(int) * stride_x
    slide_tiles["y"] = (
        slide_tiles["y"] * slide["mpp_y"] / mpp_y / stride_y
    ).round().astype(int) * stride_y

    in_bounds = (
        (slide_tiles["x"] >= 0)
        & (slide_tiles["y"] >= 0)
        & (slide_tiles["x"] + tile_extent_x <= mask_extent_x)
        & (slide_tiles["y"] + tile_extent_y <= mask_extent_y)
    )
    slide_tiles = slide_tiles.loc[in_bounds].copy()

    source_extents = np.array([mask_extent_y, mask_extent_x], dtype=np.int64)
    source_tile_extent = np.array([tile_extent_y, tile_extent_x], dtype=np.int64)
    stride = np.array([stride_y, stride_x], dtype=np.int64)

    # MaskBuilder scales the mask down if tiles overshoot the slide edge. To match
    # the old ScalarMaskBuilder behavior (and stay aligned with the outlines mask),
    # size the accumulator to the full tile span and crop back to the slide size.
    num_tiles = (
        np.ceil(np.maximum(0, source_extents - source_tile_extent) / stride).astype(
            np.int64,
        )
        + 1
    )
    span = (num_tiles - 1) * stride + source_tile_extent

    for percentage_col in [*percentage_cols]:
        filename = f"{Path(slide['path']).stem}.tiff"
        save_dir = output_path / percentage_col
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        builder = MaskBuilder(
            source_extents=tuple(span),
            source_tile_extent=tuple(source_tile_extent),
            output_tile_extent=tuple(source_tile_extent),
            stride=tuple(stride),
        )

        coords = np.stack(
            [slide_tiles["y"].to_numpy(), slide_tiles["x"].to_numpy()],
            axis=1,
        )
        data = slide_tiles[percentage_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        if len(coords) > 0:
            builder.update_batch(data, coords)

        result = builder.finalize()
        mask = result["mask"][:, :mask_extent_y, :mask_extent_x]
        mask_vips = pyvips.Image.new_from_array(mask.transpose(1, 2, 0))
        mask_vips = (mask_vips * 255).cast(pyvips.BandFormat.UCHAR)
        write_big_tiff(mask_vips, save_path, mpp_x, mpp_y)

        builder.cleanup()

    # Outlines
    mask = tile_mask(
        slide_tiles,
        tile_extent=(tile_extent_x, tile_extent_y),
        size=(mask_extent_x, mask_extent_y),
        outline_width=1,
    )

    mask_path = output_path / "outlines" / f"{Path(slide['path']).stem}.tiff"
    write_big_tiff(
        pyvips.Image.new_from_array(np.array(mask)),
        mask_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )


@with_cli_args(["+preprocessing=tile_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_path = Path(config.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    for percentage_col in [*config.percentage_cols, "outlines"]:
        (output_path / str(percentage_col)).mkdir(parents=True, exist_ok=True)

    for uri in config.dataset.mlflow_uris.tiling_filtered.values():
        local_path = Path(mlflow.artifacts.download_artifacts(uri))

        slides = pd.read_parquet(local_path / "slides")
        tiles = pd.read_parquet(local_path / "tiles")

        process_items(
            (slide for _, slide in slides.iterrows()),
            process_slide,
            fn_kwargs={
                "percentage_cols": config.percentage_cols,
                "output_path": output_path,
                "tiles": tiles,
                "level": config.level,
            },
            max_concurrent=config.max_concurrent,
        )

    logger.log_artifacts(str(output_path), artifact_path=config.mlflow_artifact_path)


if __name__ == "__main__":
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):  # type: ignore[call-arg]
        main()
