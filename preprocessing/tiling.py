import json
import tempfile
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


class _RayCpuResources(TypedDict):
    num_cpus: float


class _RayMemResources(TypedDict):
    memory: int


LO_CPU: _RayCpuResources = {"num_cpus": 0.1}
HI_CPU: _RayCpuResources = {"num_cpus": 0.2}
LO_MEM: _RayMemResources = {"memory": 128 * 1024**2}
HI_MEM: _RayMemResources = {"memory": 256 * 1024**2}


def qc_agg(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    qc_df = cast("pd.Series", df.loc[Path(row["path"]).stem])

    row["blur_mean"] = qc_df[QC_BLUR_MEAN_COLUMN]
    row["artifacts_mean"] = qc_df[QC_ARTIFACTS_MEAN_COLUMN]

    return row


def add_fold(row: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    row["fold"] = df.loc[Path(row["path"]).stem, "fold"]
    return row


def add_mask_paths(
    row: dict[str, Any], qc_folder: Path, tissue_folder: Path, annot_folder: Path
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


def tile(row: dict[str, Any], full_roi: BaseGeometry) -> list[dict[str, Any]]:
    coords_gen = grid_tiles(
        slide_extent=(row["extent_x"], row["extent_y"]),
        tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
        stride=(row["stride_x"], row["stride_y"]),
    )

    coords_list = list(coords_gen)
    downsample = row["downsample"]
    tile_area = full_roi.area

    lg_geoms = [
        geom
        for geom, label in zip(
            row["annotation_geoms"], row["annotation_labels"], strict=True
        )
        if label == "LG Dysplasia"
    ]
    hg_geoms = [
        geom
        for geom, label in zip(
            row["annotation_geoms"], row["annotation_labels"], strict=True
        )
        if label == "HG Dysplasia" or label == "Loose HG Dysplasia"
    ]

    lg_gen = tile_annotations(lg_geoms, full_roi, coords_list, downsample)
    hg_gen = tile_annotations(hg_geoms, full_roi, coords_list, downsample)

    return [
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
            "tissue_mask_path": row["tissue_mask_path"],
            "blur_mask_path": row["blur_mask_path"],
            "artifacts_mask_path": row["artifacts_mask_path"],
            "LG Dysplasia": lg_p.area / tile_area,
            "HG Dysplasia": hg_p.area / tile_area,
            "annotation": (lg_p.area + hg_p.area) / tile_area,
        }
        for coord, lg_p, hg_p in zip(coords_list, lg_gen, hg_gen, strict=True)
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


def load_slide_annotations(row: dict[str, Any], annot_folder: Path) -> dict[str, Any]:
    slide_path = Path(row["path"])
    slide_name = slide_path.stem
    annot_path = annot_folder / f"{slide_name}.json"

    with open(annot_path) as f:
        data = json.load(f)

    geoms = []
    labels = []

    for item in data.get("items", []):
        coords = item.get("coordinates")

        if coords and len(coords) >= 3:
            poly = Polygon(coords)

            if not poly.is_valid:
                poly = poly.buffer(0)

            geoms.append(poly)
            labels.append(item.get("name", "Unknown"))

    if geoms:
        row["annotation_geoms"] = geoms
        row["annotation_labels"] = labels

    return row


def filter_tissue(row: dict[str, Any], threshold: float) -> bool:
    return row["tissue"] >= threshold


def select(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "slide_id": row["slide_id"],
        "x": row["tile_x"],
        "y": row["tile_y"],
        "tissue": row["tissue"],
        "annotation": row["annotation"],
        "LG Dysplasia": row["LG Dysplasia"],
        "HG Dysplasia": row["HG Dysplasia"],
        "blur": row["blur"],
        "artifacts": row["artifacts"],
    }


def tiling(
    df: pd.DataFrame,
    qc_folder: Path,
    tissue_folder: Path,
    annot_folder: Path,
    tile_extent: int,
    stride: int,
    mpp: float,
    tissue_threshold: float,
    target_groups: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_df = pd.read_csv(qc_folder / "qc_metrics.csv", index_col="slide_name")
    paths = df["slide_path"].tolist()

    slides = (
        read_slides(paths, tile_extent=tile_extent, stride=stride, mpp=mpp)
        .map(row_hash, **LO_CPU, **LO_MEM)
        .map(qc_agg, fn_args=(qc_df,), **HI_CPU, **LO_MEM)  # pyright: ignore[reportArgumentType]
        .map(load_slide_annotations, fn_args=(annot_folder,), **LO_CPU, **HI_MEM)
    )

    if "fold" in df.columns:
        slides = slides.map(add_fold, fn_args=(df,), **LO_CPU, **LO_MEM)  # pyright: ignore[reportArgumentType]

    tissue_roi = create_tissue_roi(tile_extent)
    full_roi = create_full_roi(tile_extent)

    tiles = (
        slides.map(
            add_mask_paths,  # pyright: ignore[reportArgumentType]
            fn_args=(qc_folder, tissue_folder, annot_folder),
            **LO_CPU,
            **LO_MEM,
        )
        .flat_map(tile, fn_args=(full_roi,), **HI_CPU, **LO_MEM)
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
            **HI_CPU,
            **HI_MEM,
        )
        .map(extract_coverages, fn_args=("tissue",), **LO_CPU, **LO_MEM)  # pyright: ignore[reportArgumentType]
        .filter(filter_tissue, fn_args=(tissue_threshold,), **LO_CPU, **LO_MEM)  # pyright: ignore[reportArgumentType]
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
            **HI_CPU,
            **HI_MEM,
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
            **HI_CPU,
            **HI_MEM,
        )
        .map(extract_coverages, fn_args=("blur", "artifacts"), **LO_CPU, **LO_MEM)  # pyright: ignore[reportArgumentType]
        .map(select, **LO_CPU, **LO_MEM)
    )

    slides = slides.drop_columns(["annotation_geoms", "annotation_labels"])

    return slides.to_pandas(), tiles.to_pandas()


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    qc_folder = Path(download_artifacts(config.mlflow_uris.qc_uri))
    tissue_folder = Path(download_artifacts(config.mlflow_uris.tissue_uri))
    annot_folder = Path(config.annot_path)

    for name, split_uri in config.mlflow_uris.splits.items():
        split = pd.read_csv(
            mlflow.artifacts.download_artifacts(split_uri), index_col="slide_id"
        )

        df_slides, df_tiles = tiling(
            split,
            qc_folder=qc_folder,
            tissue_folder=tissue_folder,
            annot_folder=annot_folder,
            tile_extent=config.tile_extent,
            stride=config.stride,
            mpp=config.mpp,
            tissue_threshold=config.tissue_threshold,
            target_groups=config.target_groups,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / name
            save_dir.mkdir(parents=True, exist_ok=True)
            df_slides.to_parquet(save_dir / "slides.parquet", index=False)
            df_tiles.to_parquet(save_dir / "tiles.parquet", index=False)

            mlflow.log_artifacts(tmpdir, config.mlflow_artifact_path)


if __name__ == "__main__":
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
