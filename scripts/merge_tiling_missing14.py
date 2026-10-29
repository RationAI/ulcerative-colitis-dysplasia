#!/usr/bin/env python
"""Merge preprocessing/tiling_missing14 into a NEW preprocessing/tiling_200.

The 186-slide tiling's tile shards are copied byte-for-byte (shutil.copy2) and
the new 14-slide shards are appended as the next shard numbers; slides.parquet
is rewritten with all 200 rows. Nothing under preprocessing/tiling is touched,
so the existing tiling stays intact for whatever still points at it.

Point configs at the result afterwards:
  tiling_dir: ${project_path}/preprocessing/tiling_200

Usage (from the notebook pod):
    .venv/bin/python scripts/merge_tiling_missing14.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
TILING = PROJECT / "preprocessing" / "tiling"
NEW = PROJECT / "preprocessing" / "tiling_missing14"
OUT = PROJECT / "preprocessing" / "tiling_200"


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} already exists -- refusing to merge into it")

    (OUT / "slides").mkdir(parents=True)
    (OUT / "tiles").mkdir()

    old_slides = pd.read_parquet(TILING / "slides" / "slides.parquet")
    new_slide_files = sorted(NEW.glob("slides/*.parquet"))
    if not new_slide_files:
        sys.exit(f"no per-slide slides parquets in {NEW / 'slides'}")
    new_slides = pd.concat(
        (pd.read_parquet(f) for f in new_slide_files), ignore_index=True
    )
    overlap = set(old_slides["path"]) & set(new_slides["path"])
    if overlap:
        sys.exit(f"slides in both tilings: {sorted(overlap)}")
    if list(new_slides.columns) != list(old_slides.columns):
        sys.exit(
            f"slides columns differ:\n{list(new_slides.columns)}\nvs\n{list(old_slides.columns)}"
        )
    merged_slides = pd.concat([old_slides, new_slides], ignore_index=True)
    merged_slides.to_parquet(OUT / "slides" / "slides.parquet", index=False)
    print(
        f"slides: {len(old_slides)} + {len(new_slides)} = {len(merged_slides)} "
        f"-> {merged_slides['split'].value_counts().to_dict()}"
    )

    #: Column check on the data, then byte-for-byte copy of the old shards.
    old_cols = pq.read_schema(sorted(TILING.glob("tiles/*.parquet"))[0]).names
    old_files = sorted(TILING.glob("tiles/*.parquet"))
    new_files = sorted(NEW.glob("tiles/*.parquet"))
    for f in new_files:
        if pq.read_schema(f).names != old_cols:
            sys.exit(f"{f.name} columns differ from the existing tiling")

    offset = len(old_files)
    total = 0
    for i, f in enumerate(old_files):
        shutil.copy2(f, OUT / "tiles" / f.name)
        total += pq.ParquetFile(f).metadata.num_rows
    for j, f in enumerate(new_files):
        dest = OUT / "tiles" / f"tiles_{offset + j:04d}.parquet"
        shutil.copy2(f, dest)
        total += pq.ParquetFile(f).metadata.num_rows
        print(f"  {f.name} -> {dest.name}")

    # No duplicate (slide_id, x, y) across the merged shards.
    keys = []
    for f in sorted((OUT / "tiles").glob("*.parquet")):
        t = pd.read_parquet(f, columns=["slide_id", "x", "y"])
        keys.append(t)
    keys = pd.concat(keys, ignore_index=True)
    dups = int(keys.duplicated().sum())
    if dups:
        sys.exit(f"{dups} duplicate (slide_id, x, y) rows in {OUT}")

    print(f"\ntiles: {total:,} across {len(old_files) + len(new_files)} files -> {OUT}")
    print(f"splits: {merged_slides['split'].value_counts().to_dict()}")
    print("\npoint tiling_dir at preprocessing/tiling_200")


if __name__ == "__main__":
    main()
