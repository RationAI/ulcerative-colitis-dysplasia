import re
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypedDict, cast

import hydra
import mlflow
import pandas as pd
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.parsers.empaia_parser import EMPAIAParser
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, tile_annotations, tile_overlay_overlap
from ratiopath.tiling.utils import row_hash
from ray.data.expressions import col
from shapely import Polygon
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


QC_BLUR_MEAN_COLUMN = "mean_coverage(Piqe)"
QC_ARTIFACTS_MEAN_COLUMN = "mean_coverage(ResidualArtifactsAndCoverage)"
QC_SUBFOLDERS = {"blur": "blur_per_pixel", "artifacts": "artifacts_per_pixel"}


def qc_agg(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    qc_df = cast("pd.Series", df.loc[Path(row["path"]).stem])

    row["blur_mean"] = qc_df[QC_BLUR_MEAN_COLUMN]
    row["artifacts_mean"] = qc_df[QC_ARTIFACTS_MEAN_COLUMN]

    return row


def add_fold(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    row["fold"] = df.loc[Path(row["path"]).stem, "fold"]
    return row


def add_annot_path(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    row["annot_path"] = df.loc[Path(row["path"]).stem, "annot_path"]
    return row


def add_clarity(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    row["clarity"] = df.loc[Path(row["path"]).stem, "clarity"]
    return row


def add_mask_paths(
    row: dict[str, Any],
    qc_folder: Path,
    tissue_folder: Path,
) -> dict[str, Any]:
    stem = Path(row["path"]).stem
    row["tissue_mask_path"] = str(tissue_folder / f"{stem}.tiff")
    for key, subfolder in QC_SUBFOLDERS.items():
        row[f"{key}_mask_path"] = str(qc_folder / subfolder / f"{stem}.tiff")

    return row


def create_tissue_roi(tile_extent: int) -> Polygon:
    offset = tile_extent // 4
    size = tile_extent // 2
    return box(offset, offset, offset + size, offset + size)


def create_full_roi(tile_extent: int) -> Polygon:
    return box(0, 0, tile_extent, tile_extent)


def generate_generators(
    annot_path: Path,
    full_roi: BaseGeometry,
    coords_list: list[tuple[int, int]],
    downsample: float,
    target_groups: list[str],
) -> list[Generator[Any, None, None]]:
    class_generators = []
    parser = EMPAIAParser(annot_path)
    for label in target_groups:
        polygons = [
            p
            for p in parser.get_polygons(name=rf"^{re.escape(label)}$")
            if not p.is_empty and p.area > 0
        ]
        if not polygons:
            class_generators.append(Polygon() for _ in coords_list)
        else:
            class_generators.append(
                tile_annotations(polygons, full_roi, coords_list, downsample)
            )

    return class_generators


def tile(
    row: dict[str, Any],
    full_roi: BaseGeometry,
    annot_folder: Path,
    target_groups: list[str],
) -> list[dict[str, Any]]:
    coords_gen = grid_tiles(
        slide_extent=(row["extent_x"], row["extent_y"]),
        tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
        stride=(row["stride_x"], row["stride_y"]),
    )

    coords_list = list(coords_gen)
    downsample = row["downsample"]
    tile_area = full_roi.area

    slide_name = Path(row["path"]).stem
    annot_path = annot_folder / f"{slide_name}.json"

    class_generators = []
    if row["annot_path"] == "NEGATIVE":
        for _ in target_groups:
            class_generators.append(Polygon() for _ in coords_list)
    else:
        class_generators = generate_generators(
            annot_path, full_roi, coords_list, downsample, target_groups
        )

    return [
        (
            {
                "tile_x": coord[0],
                "tile_y": coord[1],
                "path": row["path"],
                "slide_id": row["id"],
                "level": row["level"],
                "tile_extent_x": row["tile_extent_x"],
                "tile_extent_y": row["tile_extent_y"],
                "mpp_x": row["mpp_x"],
                "mpp_y": row["mpp_y"],
                "clarity": row.get("clarity"),
                "fold": row.get("fold"),
                "tissue_mask_path": row["tissue_mask_path"],
                "blur_mask_path": row["blur_mask_path"],
                "artifacts_mask_path": row["artifacts_mask_path"],
                **{
                    label: class_poly.area / tile_area
                    for label, class_poly in zip(
                        target_groups, class_polys, strict=True
                    )
                },
                "annotation": sum(class_poly.area for class_poly in class_polys)
                / tile_area,
            }
        )
        for coord, *class_polys in zip(coords_list, *class_generators, strict=True)
    ]


def extract_coverages(row: dict[str, Any], *cols: str) -> dict[str, Any]:
    for c in cols:
        overlap = row[f"{c}_overlap"]
        zero_overlap = overlap.get("0", 0)
        if zero_overlap is None:
            row[c] = 1.0
        else:
            row[c] = 1.0 - zero_overlap

    return row


def filter_tissue(row: dict[str, Any], threshold: float) -> bool:
    return row["tissue"] >= threshold


def select(row: dict[str, Any], target_labels: list[str]) -> dict[str, Any]:
    selected_row = {
        "slide_id": row["slide_id"],
        "x": row["tile_x"],
        "y": row["tile_y"],
        "tissue": row["tissue"],
        "annotation": row["annotation"],
        "blur": row["blur"],
        "artifacts": row["artifacts"],
        "clarity": row.get("clarity"),
        "fold": row.get("fold"),
    }

    for label in target_labels:
        selected_row[label] = row[label]

    return selected_row


def tiling(
    df: pd.DataFrame,
    qc_folder: Path,
    tissue_folder: Path,
    annot_folder: Path,
    tile_extent: int,
    stride: int,
    mpp: float,
    tissue_threshold: float,
    target_groups: list[str],
) -> tuple[ray.data.Dataset, ray.data.Dataset]:
    qc_df = pd.read_csv(qc_folder / "qc_metrics.csv", index_col="slide_name")

    paths = df["slide_path"].tolist()

    slides = (
        read_slides(paths, tile_extent=tile_extent, stride=stride, mpp=mpp)
        .map(row_hash)
        .map(add_annot_path, fn_args=(df,))
        .map(qc_agg, fn_args=(qc_df,))  # pyright: ignore[reportArgumentType]
    )

    if "fold" in df.columns:
        slides = slides.map(add_fold, fn_args=(df,))  # pyright: ignore[reportArgumentType]

    if "clarity" in df.columns:
        slides = slides.map(add_clarity, fn_args=(df,))  # pyright: ignore[reportArgumentType]

    tissue_roi = create_tissue_roi(tile_extent)
    full_roi = create_full_roi(tile_extent)

    tiles = (
        slides.map(
            add_mask_paths,  # pyright: ignore[reportArgumentType]
            fn_args=(qc_folder, tissue_folder),
        )
        .flat_map(
            tile,
            fn_args=(full_roi, annot_folder, target_groups),
        )
        .repartition(target_num_rows_per_block=4096)
        .with_column(
            "tissue_overlap",
            tile_overlay_overlap(
                tissue_roi,
                col("tissue_mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),  # pyright: ignore[reportCallIssue]
        )
        .map(extract_coverages, fn_args=("tissue",))  # pyright: ignore[reportArgumentType]
        .filter(filter_tissue, fn_args=(tissue_threshold,))  # pyright: ignore[reportArgumentType]
        .with_column(
            "blur_overlap",
            tile_overlay_overlap(
                full_roi,
                col("blur_mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),  # pyright: ignore[reportCallIssue]
        )
        .with_column(
            "artifacts_overlap",
            tile_overlay_overlap(
                full_roi,
                col("artifacts_mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),  # pyright: ignore[reportCallIssue]
        )
        .map(extract_coverages, fn_args=("blur", "artifacts"))  # pyright: ignore[reportArgumentType]
        .map(select, fn_args=(target_groups,))  # pyright: ignore[reportArgumentType]
    )

    return slides, tiles


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    qc_folder = Path(download_artifacts(config.mlflow_uris.qc))
    tissue_folder = Path(download_artifacts(config.mlflow_uris.tissue))
    annot_folder = Path(config.annot_path)

    for name, split_uri in config.mlflow_uris.splits.items():
        split = pd.read_csv(
            mlflow.artifacts.download_artifacts(split_uri), index_col="slide_id"
        )

        ds_slides, ds_tiles = tiling(
            split,
            qc_folder=qc_folder,
            tissue_folder=tissue_folder,
            annot_folder=annot_folder,
            tile_extent=config.tile_extent,
            stride=config.stride,
            mpp=config.mpp,
            tissue_threshold=config.tissue_threshold,
            target_groups=list(config.target_groups),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / name
            save_dir.mkdir(parents=True, exist_ok=True)

            slides_pd = ds_slides.to_pandas()
            tiles_pd = ds_tiles.to_pandas()

            slides_pd.to_parquet(save_dir / "slides.parquet", index=False)
            tiles_pd.to_parquet(save_dir / "tiles.parquet", index=False)

            mlflow.log_artifacts(tmpdir, config.mlflow_artifact_path)


if __name__ == "__main__":
    ray.init(runtime_env={"excludes": [".git", ".venv"]})
    try:
        main()
    finally:
        ray.shutdown()
