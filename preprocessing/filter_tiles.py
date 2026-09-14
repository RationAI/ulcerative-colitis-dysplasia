import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def mark_slide_origin(group: pd.DataFrame) -> pd.DataFrame:
    """Add a ``from_negative_slide`` flag marking tiles from purely negative slides.

    A slide is treated as purely negative when its total annotation coverage is
    zero. All tiles from such a slide keep ``from_negative_slide = True``;
    tiles from slides that carry any annotation get ``from_negative_slide =
    False``.

    Args:
        group: A Pandas DataFrame containing all tiles for a specific
            ``slide_id``. Must include an ``annotation`` column.

    Returns:
        A copy of ``group`` with the new ``from_negative_slide`` boolean column.
    """
    from_negative_slide = float(group["annotation"].sum()) == 0.0
    group = group.copy()
    group["from_negative_slide"] = from_negative_slide
    return group


def filter_slide_tiles_annotated_column(group: pd.DataFrame) -> pd.DataFrame:
    """Filters tiles within a single slide to keep only the tissue section (column)
    containing annotations.

    Whole Slide Images (WSIs) in this dataset often contain a grid of multiple
    tissue sections (e.g., a 3x4 grid creating 3 distinct vertical columns).
    Pathologists typically only annotate one of these columns. Tiles from
    unannotated sections may appear negative but are actually "unknown" and
    should be discarded to prevent false negatives.

    The function identifies tissue columns by clustering tiles based on their
    x-coordinates using a dynamic gap threshold.

    Logic:
    1. If the slide has zero total annotations, all tiles are kept (treated as
       a true negative slide).
    2. Clusters are identified by finding large jumps in the x-axis (gaps > 50%
       of the maximum observed gap).
    3. If annotations exist, the function identifies which cluster(s) they fall
       into.
    4. If annotations span more than one cluster, a warning is logged.
    5. Only tiles belonging to the annotated cluster are returned.

    Args:
        group: A Pandas DataFrame containing all tiles for a specific
            ``slide_id``. Must include ``x``, ``annotation``, and ``slide_id``
            columns.

    Returns:
        A DataFrame containing only the tiles from the annotated tissue
        column, with the temporary cluster ID removed.
    """  # noqa: D205
    if group["annotation"].sum() == 0:
        return mark_slide_origin(group)

    sorted_group = group.sort_values("x").copy()

    unique_x = sorted_group["x"].drop_duplicates().sort_values()
    x_diffs = unique_x.diff()

    dynamic_gap_threshold = x_diffs.max() * 0.50
    clusters = (x_diffs > dynamic_gap_threshold).cumsum().fillna(0)
    x_to_cluster = dict(zip(unique_x, clusters, strict=True))

    sorted_group["_cluster"] = sorted_group["x"].map(x_to_cluster)

    valid_clusters = sorted_group.groupby("_cluster")["annotation"].sum()
    valid_ids = valid_clusters[valid_clusters > 0].index

    filtered = sorted_group[sorted_group["_cluster"].isin(valid_ids)]

    return mark_slide_origin(filtered.drop(columns=["_cluster"]))


@with_cli_args(["+preprocessing=filter_tiles"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        for split, split_uri in config.dataset.mlflow_uris.tiling.items():
            local_dir = Path(mlflow.artifacts.download_artifacts(split_uri))

            slides = local_dir / "slides"
            tiles = local_dir / "tiles"

            ds_tiles = ray.data.read_parquet(str(tiles))
            ds_slides = ray.data.read_parquet(str(slides))

            ds_slides = ds_slides.map_batches(
                lambda df: df[["id", "path"]].rename(columns={"id": "slide_id"}),
                batch_format="pandas",
            )

            ds_tiles_with_path = ds_tiles.join(
                ds_slides, join_type="inner", num_partitions=2, on=["slide_id"]
            )
            filtered_ds_tiles = ds_tiles_with_path.groupby("slide_id").map_groups(
                filter_slide_tiles_annotated_column, batch_format="pandas"
            )

            save_dir = Path(tmpdir) / split
            save_dir.mkdir(parents=True, exist_ok=True)

            ds_slides = ray.data.read_parquet(str(slides))
            rows = config.row_per_file
            ds_slides.write_parquet(str(save_dir / "slides"), min_rows_per_file=rows)
            filtered_ds_tiles.write_parquet(
                str(save_dir / "tiles"), min_rows_per_file=rows
            )

        logger.log_artifacts(tmpdir, config.mlflow_artifact_path)


if __name__ == "__main__":
    ray.init(runtime_env={"excludes": [".git", ".venv"]})
    try:
        main()
    finally:
        ray.shutdown()
