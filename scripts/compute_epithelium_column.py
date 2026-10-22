#!/usr/bin/env python
"""Compute the real `epithelium` column for tiling_186_v3 from the epi-200 masks.

The v3 run predated the finished epi masks, so its `epithelium` column is a
stand-in copy of `tissue` (epithelium_folder fell back to tissue_masks). This
re-derives it properly WITHOUT re-tiling: for each slide it loads the epi mask's
level-2 top page (147200x136384, the tiles' own resolution), binarizes at 0.5
probability, and uses an integral image to get each tile's epithelium fraction
in O(1). No whole-slide opens -- the part that made tiling take 3 h -- so this
runs in minutes per slide at 4 workers.

The epi masks are probability heatmaps (0-255), not binary segmentations, so a
threshold is needed; 128 (=0.5 probability) matches the tissue_threshold: 0.5
convention the tissue column uses.

Output: one parquet, columns (slide_id, x, y, epithelium) -- the join table the
embeddings step merges on (slide_id, x, y). Per-slide part files under the
parts dir make a re-run resume.

Usage (from the notebook pod):
    .venv/bin/python -I scripts/compute_epithelium_column.py
"""
import concurrent.futures
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tifffile import TiffFile

TILE_BASE = "/mnt/projects/huncrc_dysplasia/tiling_186_v3"
EPI_DIR = "/mnt/projects/huncrc_dysplasia/epithelium_masks/raw"
PARTS_DIR = "/mnt/projects/huncrc_dysplasia/epithelium_column_parts"
OUT = "/mnt/projects/huncrc_dysplasia/epithelium_column_186.parquet"
TILE = 224
#: 0.5 probability on the 0-255 heatmap -- same 0.5 convention as tissue_threshold.
EPI_THRESHOLD = 128
WORKERS = 4

EXPECTED_SHAPE = (147200, 136384)  # epi top page = slide level 2 (H, W)


def load_tile_table() -> tuple[pd.DataFrame, dict[str, str]]:
    sd = pd.concat(
        [pq.read_table(f"{TILE_BASE}/shard{s}/slides/slides.parquet").to_pandas()
         for s in (1, 2, 3)], ignore_index=True)
    h2n = dict(zip(sd["id"], sd["path"].str.extract(r"(\d{3})\.mrxs$")[0]))
    g = None
    for s in (1, 2, 3):
        import glob
        files = sorted(glob.glob(f"{TILE_BASE}/shard{s}/tiles/*.parquet"))
        t = pd.concat([pq.read_table(f, columns=["slide_id", "x", "y"]).to_pandas()
                       for f in files], ignore_index=True)
        g = t if g is None else pd.concat([g, t], ignore_index=True)
    return g, h2n


def epi_fraction(num: str) -> pd.DataFrame:
    """Per-tile epithelium fraction for one slide, via integral image."""
    import os

    out_path = f"{PARTS_DIR}/{num}.parquet"
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)

    with TiffFile(f"{EPI_DIR}/{num}.tiff") as tf:
        assert tf.pages[0].shape == EXPECTED_SHAPE, \
            f"{num}: epi top page {tf.pages[0].shape}, expected {EXPECTED_SHAPE}"
        m = tf.asarray(0)
    binary = (m >= EPI_THRESHOLD).astype(np.float64)
    del m
    # Integral image with a zero top row and left column for O(1) window sums.
    S = np.pad(binary.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    del binary

    tiles = pd.read_parquet(f"{PARTS_DIR}/__index__/{num}.parquet")
    x, y = tiles["x"].to_numpy(), tiles["y"].to_numpy()
    bad = (x + TILE > EXPECTED_SHAPE[1]) | (y + TILE > EXPECTED_SHAPE[0])
    assert not bad.any(), f"{num}: {int(bad.sum())} tiles extend past the mask extent"
    frac = (S[y + TILE, x + TILE] - S[y, x + TILE] - S[y + TILE, x] + S[y, x]) / (TILE * TILE)
    out = tiles.assign(epithelium=frac)
    assert out["epithelium"].between(0.0, 1.0).all()
    out.to_parquet(out_path, index=False)
    return out


def main() -> None:
    import os

    os.makedirs(f"{PARTS_DIR}/__index__", exist_ok=True)
    tiles, h2n = load_tile_table()
    tiles["num"] = tiles["slide_id"].map(h2n)
    assert tiles["num"].notna().all()
    print(f"{len(tiles):,} tiles, {tiles['num'].nunique()} slides")

    # Per-slide (slide_id, x, y) index files for the workers.
    for num, grp in tiles.groupby("num", sort=True):
        part = f"{PARTS_DIR}/__index__/{num}.parquet"
        if not os.path.exists(part):
            grp[["slide_id", "x", "y"]].to_parquet(part, index=False)

    nums = sorted(tiles["num"].unique())
    parts = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(epi_fraction, num): num for num in nums}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            num = futs[fut]
            part = fut.result()  # raises on worker failure
            parts.append(part)
            print(f"[{i}/{len(nums)}] {num}: {len(part)} tiles, "
                  f"epi mean {part['epithelium'].mean():.3f}", flush=True)

    out = pd.concat(parts, ignore_index=True)
    out = out[["slide_id", "x", "y", "epithelium"]]
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} rows "
          f"(expect {len(tiles):,}), epi mean {out['epithelium'].mean():.3f} "
          f"median {out['epithelium'].median():.3f} "
          f"zero-epi tiles {int((out['epithelium'] == 0).sum()):,}")


if __name__ == "__main__":
    main()
