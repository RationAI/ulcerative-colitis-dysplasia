"""IBD-34 Step 4: fold-0 prediction heatmaps (CPU-only).

Renders the four level-1 ``high`` configs' fold-0 val predictions as continuous
``prob_high`` heatmaps, aligned with the existing level-3 epithelium /
annotation masks.

Unlike Steps 1-3 there is no forward pass, no model load and no tile
processing: each config's predictions were already logged by the Step-1 valfold
job as ``val_predictions.parquet`` (``slide_name`` / ``x`` / ``y`` /
``prob_high``). Those tile coordinates already live in the annotated-column
grid used to build the level-3 masks on the ``feat/tile_masks`` branch, so the
predictions are rendered **directly** with the same
``ratiopath.masks.MaskBuilder`` level-3 path -- no join to the tiles parquet is
needed. The slides parquet only supplies per-slide geometry (path / mpp /
tile extent / stride) for the coordinate conversion, and for opening the WSI.

Heatmaps are rasterized per slide (de-duplicated on ``id``) with the level-3
renderer ported from ``preprocessing/tile_masks.py`` and logged per config.

The manifest has one row per fold-0 training run; ``epi`` / ``negative_slides``
name the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import pandas as pd
import pyvips
from datasets import Dataset as HFDataset
from omegaconf import DictConfig, OmegaConf
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder
from ratiopath.openslide import OpenSlide

from ml.data.datasets.labels import process_slides


if TYPE_CHECKING:
    from rationai.mlkit.lightning.loggers import MLFlowLogger


# ---------------------------------------------------------------------------
# Level-3 rendering (ported from preprocessing/tile_masks.py on feat/tile_masks)
# ---------------------------------------------------------------------------


def _render_heatmap(
    tiles,
    slide,
    value_col: str,
    save_path: Path,
    level: int,
) -> Path:
    """Rasterize a continuous value column for one slide at ``level``.

    Mirrors ``tile_masks.process_slide``: tile coordinates are converted to the
    target level via the MPP ratio and stride-snapped, and the whole-slide
    ``MaskBuilder`` accumulator is built at once (then cropped) so there is no
    per-tile scale-down misalignment.
    """
    import pandas as pd

    tiles = pd.DataFrame(tiles)

    with OpenSlide(slide["path"]) as slide_wsi:
        mask_extent_x, mask_extent_y = slide_wsi.level_dimensions[level]
        mpp_x, mpp_y = slide_wsi.slide_resolution(level)

    tile_extent_x = int(slide["tile_extent_x"] / mpp_x * slide["mpp_x"])
    tile_extent_y = int(slide["tile_extent_y"] / mpp_y * slide["mpp_y"])
    stride_x = int(slide["stride_x"] / mpp_x * slide["mpp_x"])
    stride_y = int(slide["stride_y"] / mpp_y * slide["mpp_y"])

    tiles["x"] = (tiles["x"] * slide["mpp_x"] / mpp_x / stride_x).round().astype(
        int
    ) * stride_x
    tiles["y"] = (tiles["y"] * slide["mpp_y"] / mpp_y / stride_y).round().astype(
        int
    ) * stride_y

    in_bounds = (
        (tiles["x"] >= 0)
        & (tiles["y"] >= 0)
        & (tiles["x"] + tile_extent_x <= mask_extent_x)
        & (tiles["y"] + tile_extent_y <= mask_extent_y)
    )
    tiles = tiles.loc[in_bounds]

    source_extents = np.array([mask_extent_y, mask_extent_x], dtype=np.int64)
    source_tile_extent = np.array([tile_extent_y, tile_extent_x], dtype=np.int64)
    stride = np.array([stride_y, stride_x], dtype=np.int64)
    num_tiles = (
        np.ceil(np.maximum(0, source_extents - source_tile_extent) / stride).astype(
            np.int64
        )
        + 1
    )
    span = (num_tiles - 1) * stride + source_tile_extent

    builder = MaskBuilder(
        source_extents=tuple(span),
        source_tile_extent=tuple(source_tile_extent),
        output_tile_extent=tuple(source_tile_extent),
        stride=tuple(stride),
    )

    coords = np.stack([tiles["y"].to_numpy(), tiles["x"].to_numpy()], axis=1)
    data = tiles[value_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    if len(coords) > 0:
        builder.update_batch(data, coords)

    result = builder.finalize()
    mask = result["mask"][:, :mask_extent_y, :mask_extent_x]

    mask_vips = pyvips.Image.new_from_array(mask.transpose(1, 2, 0))
    mask_vips = (mask_vips * 255).cast(pyvips.BandFormat.UCHAR)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_big_tiff(mask_vips, save_path, mpp_x, mpp_y)
    builder.cleanup()
    return save_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _dedup_slides(slides) -> HFDataset:
    """Drop duplicate rows so each slide is rendered exactly once."""
    seen: set[str] = set()
    keep: list[int] = []
    for idx, s in enumerate(slides):
        if s["id"] not in seen:
            seen.add(s["id"])
            keep.append(idx)
    return slides.select(keep)


def _load_fold0_slides(uri: str):
    """Load fold-0 slide geometry from the filtered-tiles slides dir."""
    local = mlflow.artifacts.download_artifacts(uri)
    parquets = sorted(str(p) for p in Path(local).iterdir() if p.suffix == ".parquet")
    if not parquets:
        parquets = sorted(str(p) for p in Path(local).glob("*.parquet"))
    slides = HFDataset.from_parquet(parquets)
    slides = process_slides(slides, val_fold=0, is_val=True)
    return _dedup_slides(slides)


def _load_prediction_parquet(uri: str) -> pd.DataFrame:
    local = mlflow.artifacts.download_artifacts(uri)
    if Path(local).is_dir():
        parquets = sorted(
            str(p) for p in Path(local).iterdir() if p.suffix == ".parquet"
        )
        return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    return pd.read_parquet(local)


def _prediction_parquet_uri(valfold_run_id: str, train_id: str) -> str:
    return f"runs:/{valfold_run_id}/valfold/{train_id}/val_predictions.parquet"


def _config_key(row) -> str:
    epi = float(row.get("epi", 0.0))
    neg = bool(row.negative_slides)
    return f"epi{epi:g}_neg{'on' if neg else 'off'}"


def heatmap_report_main(config: DictConfig, logger: MLFlowLogger) -> None:
    valfold_run_id = str(config.heatmap.valfold_run_id)
    level = int(config.heatmap.level)
    manifest_path = config.heatmap.manifest
    rows = OmegaConf.load(manifest_path)
    if not isinstance(rows, (list, tuple)):
        rows = rows.runs
    print(
        f"Rendering fold-0 heatmaps (level {level}) for {len(rows)} config(s) "
        f"from {manifest_path} using val predictions from run {valfold_run_id}.",
        flush=True,
    )

    train_uri = str(config.dataset.mlflow_uris.tiling_filtered.train)
    slides = _load_fold0_slides(train_uri + "/slides")
    print(f"Fold-0 val slides: {len(slides)}.", flush=True)

    run = logger.experiment.get_run(logger.run_id)
    for row in rows:
        train_id = str(row.run_id)
        config_key = _config_key(row)

        preds = _load_prediction_parquet(
            _prediction_parquet_uri(valfold_run_id, train_id)
        )
        preds = preds.drop_duplicates(subset=["slide_name", "x", "y"])
        print(f"[{train_id}] config={config_key}: {len(preds)} preds.", flush=True)

        save_dir = f"heatmaps/{config_key}"
        saved: list[str] = []
        for s in slides:
            name = s["name"]
            sub = preds.loc[preds["slide_name"] == name, ["x", "y", "prob_high"]]
            if sub.empty:
                continue
            save_path = Path(save_dir) / f"{name}.tiff"
            _render_heatmap(sub, dict(s), "prob_high", save_path, level)
            saved.append(str(save_path))

        logger.log_artifacts(save_dir, artifact_path=f"heatmaps/{config_key}")
        uri = (
            f"mlflow-artifacts:/{run.info.experiment_id}/{logger.run_id}"
            f"/artifacts/heatmaps/{config_key}"
        )
        print(
            f"[{train_id}] config={config_key} -> {len(saved)} heatmaps -> {uri}",
            flush=True,
        )
