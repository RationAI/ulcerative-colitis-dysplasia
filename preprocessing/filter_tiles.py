import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def filter_slide_tiles(group: pd.DataFrame) -> pd.DataFrame:
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
    4. If annotations span more than one cluster, a ValueError is raised as
       per the "one annotated column per slide" rule.
    5. Only tiles belonging to the annotated cluster are returned.

    Args:
        group (pd.DataFrame): A Pandas DataFrame containing all tiles for a
            specific 'slide_id'. Must include 'x', 'annotation', and 'slide_id'
            columns.

    Returns:
        pd.DataFrame: A DataFrame containing only the tiles from the annotated
            tissue column, with the temporary cluster ID removed.

    Raises:
        ValueError: If annotations are detected in multiple spatially distinct
            tissue columns on the same slide.
    """  # noqa: D205
    if group["annotation"].sum() == 0:
        return group

    sorted_group = group.sort_values("x").copy()

    unique_x = sorted_group["x"].drop_duplicates().sort_values()
    x_diffs = unique_x.diff()

    dynamic_gap_threshold = x_diffs.max() * 0.50
    clusters = (x_diffs > dynamic_gap_threshold).cumsum().fillna(0)
    x_to_cluster = dict(zip(unique_x, clusters, strict=True))

    sorted_group["_cluster"] = sorted_group["x"].map(x_to_cluster)

    valid_clusters = sorted_group.groupby("_cluster")["annotation"].sum()
    valid_ids = valid_clusters[valid_clusters > 0].index

    if len(valid_ids) > 1:
        slide_id = group["slide_id"].iloc[0]
        raise ValueError(
            f"Slide {slide_id} has annotations in {len(valid_ids)} different tissue columns. "
            "Only one column can be annotated per slide."
        )

    filtered = sorted_group[sorted_group["_cluster"].isin(valid_ids)]

    return filtered.drop(columns=["_cluster"])


@with_cli_args(["+preprocessing=filter_tiles"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_uri = config.mlflow_uris.tiling

    with tempfile.TemporaryDirectory() as tmpdir:
        for split in config.splits:
            split_uri = f"{tiling_uri}/{split}"
            local_dir = Path(mlflow.artifacts.download_artifacts(split_uri))

            slides_dir = local_dir / "slides"
            tiles_dir = local_dir / "tiles"

            ds_tiles = ray.data.read_parquet(str(tiles_dir))
            filtered_ds_tiles = ds_tiles.groupby("slide_id").map_groups(
                filter_slide_tiles, batch_format="pandas"
            )

            save_dir = Path(tmpdir) / split
            save_dir.mkdir(parents=True, exist_ok=True)

            ds_slides = ray.data.read_parquet(str(slides_dir))
            ds_slides.write_parquet(str(save_dir / "slides"))

            filtered_ds_tiles.write_parquet(
                str(save_dir / "tiles"), partition_cols=["slide_id"]
            )

        if logger is not None:
            mlflow.log_artifacts(tmpdir, config.mlflow_artifact_path)


if __name__ == "__main__":
    main()
