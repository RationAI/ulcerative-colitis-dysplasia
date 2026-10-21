#!/usr/bin/env python
"""Merge the 3-shard tiling_186_v3 into one flat slides/+tiles/ layout.

The v3 tiling job wrote shard1..3/{slides,tiles}; scripts/run_embeddings_pilot.py
(and the embeddings job) expects <tiling_dir>/slides and <tiling_dir>/tiles as
parquet dataset dirs. This concatenates them into
/mnt/projects/huncrc_dysplasia/tiling_186_merged and verifies the totals
against each shard's manifest (186 slides, 3,928,931 tiles).

Idempotent: re-running overwrites the merged files.

Usage (from the notebook pod):
    .venv/bin/python -I scripts/merge_tiling_shards.py
"""
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/mnt/projects/huncrc_dysplasia/tiling_186_v3"
OUT = "/mnt/projects/huncrc_dysplasia/tiling_186_merged"
TILES_PER_FILE = 1_000_000


def main() -> None:
    os.makedirs(f"{OUT}/slides", exist_ok=True)
    os.makedirs(f"{OUT}/tiles", exist_ok=True)

    # slides: three small files -> one
    slides = pa.concat_tables(
        [pq.read_table(f"{SRC}/shard{s}/slides/slides.parquet") for s in (1, 2, 3)]
    )
    pq.write_table(slides, f"{OUT}/slides/slides.parquet")
    print(f"slides: {slides.num_rows} rows -> {OUT}/slides/slides.parquet")
    assert slides.num_rows == 186, f"expected 186 slide rows, got {slides.num_rows}"

    # tiles: 41 files -> chunks of ~1M rows each
    total_expected = sum(
        json.load(open(f"{SRC}/shard{s}/manifest.json"))["tiles"] for s in (1, 2, 3)
    )
    files = [f for s in (1, 2, 3)
             for f in sorted(glob.glob(f"{SRC}/shard{s}/tiles/*.parquet"))]
    print(f"input tile files: {len(files)}, expected tiles: {total_expected:,}")

    buf, buf_rows, n_out, total = [], 0, 0, 0
    for f in files:
        t = pq.read_table(f)
        total += t.num_rows
        buf.append(t)
        buf_rows += t.num_rows
        if buf_rows >= TILES_PER_FILE:
            out = f"{OUT}/tiles/tiles_{n_out:04d}.parquet"
            pq.write_table(pa.concat_tables(buf), out)
            print(f"  wrote {out} ({buf_rows:,} rows)")
            buf, buf_rows, n_out = [], 0, n_out + 1
    if buf:
        out = f"{OUT}/tiles/tiles_{n_out:04d}.parquet"
        pq.write_table(pa.concat_tables(buf), out)
        print(f"  wrote {out} ({buf_rows:,} rows)")
    verdict = "OK" if total == total_expected else "MISMATCH"
    print(f"total tiles merged: {total:,} vs manifests {total_expected:,} -> {verdict}")
    assert verdict == "OK"


if __name__ == "__main__":
    main()
