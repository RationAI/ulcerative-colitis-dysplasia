from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from rationai.masks.mask_builders import ScalarMaskBuilder


def save_heatmaps(
    predictions_path: str | Path,
    slides: HFDataset,
    value_columns: list[str],
    save_dir: str | Path,
) -> list[Path]:
    """Render a per-slide scalar heatmap for each prediction column.

    The predictions parquet holds one row per tile with ``slide_name``, ``x``,
    ``y`` (top-left pixel coordinates) plus one probability column per value in
    ``value_columns``. Slide geometry (extent, mpp, tile extent, stride) comes
    from the ``slides`` dataset, which is exposed by the predict dataset
    (``MetaTiledSlides``). Overlapping tiles are averaged by
    ``ScalarMaskBuilder`` and each heatmap is saved as a pyramidal BigTIFF.

    Args:
        predictions_path: Path to the predictions parquet written by
            ``save_predictions``.
        slides: HF dataset of slide rows with ``name``, ``extent_x/y``,
            ``mpp_x/y``, ``tile_extent_x`` and ``stride_x``.
        value_columns: Prediction column names to rasterize (one heatmap each).
        save_dir: Directory to write the heatmaps into.

    Returns:
        The list of saved heatmap paths, one per slide and column, laid out as
        ``<save_dir>/<column>/<slide_name>.tiff``.
    """
    df = pd.read_parquet(predictions_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for slide in slides:
        slide_name = slide["name"]
        subset = df[df["slide_name"] == slide_name]
        if subset.empty:
            continue

        xs = torch.tensor(subset["x"].values)
        ys = torch.tensor(subset["y"].values)
        for column in value_columns:
            builder = ScalarMaskBuilder(
                save_dir=save_dir / column,
                filename=slide_name,
                extent_x=int(slide["extent_x"]),
                extent_y=int(slide["extent_y"]),
                mpp_x=float(slide["mpp_x"]),
                mpp_y=float(slide["mpp_y"]),
                extent_tile=int(slide["tile_extent_x"]),
                stride=int(slide["stride_x"]),
            )
            values = torch.tensor(subset[column].values, dtype=torch.float32).unsqueeze(
                1
            )
            builder.update(values, xs, ys)
            saved.append(builder.save())

    return saved
