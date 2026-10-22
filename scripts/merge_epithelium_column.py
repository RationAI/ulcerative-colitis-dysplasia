#!/usr/bin/env python
"""Replace the stand-in `epithelium` column in tiling_186_merged with the real
per-tile epi fraction from scripts/compute_epithelium_column.py's output.

The v3 tiling ran with tissue_folder == epithelium_folder (the tissue mask was
used as the stand-in for the epi mask), so the tiling parquet's `epithelium`
column currently holds tissue scores. This joins
/mnt/projects/huncrc_dysplasia/epithelium_column_186.parquet (slide_id, x, y,
epithelium) onto each merged tiles file on (slide_id, x, y) -- a 1:1 match,
verified 2026-10-22 (3,928,931 rows, 0 unmatched) -- and overwrites the
column in place.

Idempotent: re-running joins the same values.

Usage (from the notebook pod):
    .venv/bin/python -I scripts/merge_epithelium_column.py
"""
import glob

import pyarrow.parquet as pq

TILES_DIR = "/mnt/projects/huncrc_dysplasia/tiling_186_merged/tiles"
EPI = "/mnt/projects/huncrc_dysplasia/epithelium_column_186.parquet"


def main() -> None:
    import pandas as pd

    epi = pd.read_parquet(EPI)
    print(f"epi column: {len(epi):,} rows")

    files = sorted(glob.glob(f"{TILES_DIR}/*.parquet"))
    assert files, f"no tiles parquets under {TILES_DIR}"
    total = 0
    for f in files:
        tiles = pd.read_parquet(f)
        assert "epithelium" in tiles.columns, f"{f}: no epithelium column"
        m = tiles.merge(
            epi, on=["slide_id", "x", "y"], how="left", suffixes=("", "_epi")
        )
        assert len(m) == len(tiles), f"{f}: row count changed by join"
        assert m["epithelium_epi"].notna().all(), f"{f}: unmatched tiles"
        assert m["epithelium_epi"].between(0, 1).all(), f"{f}: fraction out of range"
        tiles["epithelium"] = m["epithelium_epi"]
        tiles.to_parquet(f)
        total += len(tiles)
        print(
            f"  {f.split('/')[-1]}: {len(tiles):,} rows, "
            f"epithelium mean {tiles['epithelium'].mean():.4f}"
        )
    print(f"merged {total:,} tiles across {len(files)} files -> {TILES_DIR}")
    assert total == len(epi), f"expected {len(epi):,} tiles, merged {total:,}"


if __name__ == "__main__":
    main()
