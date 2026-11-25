#!/usr/bin/env python
"""Build a subset tiling dir for a fast pod-local embeddings run.

The full-cohort embeddings job writes all 186 slides only at the very end
(no resume), so to start training before it finishes we embed a subset of
slides now -- straight from the verified tiling_186_merged -- into the
training consumer's layout ({project_path}/embeddings/{model}/{stem}.parquet),
which the full job's output can be added to later.

Selection is deterministic: slides WITH annotation tiles first (ranked by
annotated-tile count, so the subset is label-rich), then unannotated slides
(smallest first) until MIN_SLIDES, capped at BUDGET tiles total.

Usage (from the notebook pod):
    .venv/bin/python -I scripts/make_embedding_subset.py
"""
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

SRC = "/mnt/projects/huncrc_dysplasia/tiling_186_merged"
OUT = "/mnt/projects/huncrc_dysplasia/tiling_subset"
BUDGET = 600_000
MIN_SLIDES = 15


def main() -> None:
    tiles = pd.concat(
        [
            pd.read_parquet(f, columns=["slide_id", "annotation"])
            for f in sorted(glob.glob(f"{SRC}/tiles/*.parquet"))
        ]
    )
    slides = pd.read_parquet(f"{SRC}/slides/slides.parquet")

    per = (
        tiles.groupby("slide_id")
        .agg(n_tiles=("slide_id", "size"), n_annot=("annotation", "sum"))
        .reset_index()
    )
    annotated = per[per["n_annot"] > 0].sort_values(
        ["n_annot", "n_tiles"], ascending=[False, True]
    )
    unannotated = per[per["n_annot"] == 0].sort_values("n_tiles")

    chosen, total = [], 0
    for _, r in pd.concat([annotated, unannotated], ignore_index=True).iterrows():
        if total >= BUDGET:
            break
        if total + r["n_tiles"] > BUDGET and len(chosen) >= MIN_SLIDES:
            continue
        chosen.append(r["slide_id"])
        total += int(r["n_tiles"])
    if len(chosen) < MIN_SLIDES:
        raise SystemExit(f"only {len(chosen)} slides fit the budget")

    sel = set(chosen)
    n_annot_slides = len(sel & set(annotated["slide_id"]))
    print(f"selected {len(chosen)} slides ({n_annot_slides} with annotations), "
          f"{total:,} tiles")

    os.makedirs(f"{OUT}/slides", exist_ok=True)
    os.makedirs(f"{OUT}/tiles", exist_ok=True)

    slides_sel = slides[slides["id"].isin(sel)]
    slides_sel.to_parquet(f"{OUT}/slides/slides.parquet")

    full = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SRC}/tiles/*.parquet"))])
    full_sel = full[full["slide_id"].isin(sel)]
    full_sel.to_parquet(f"{OUT}/tiles/tiles.parquet")

    assert len(slides_sel) == len(chosen), "slide set mismatch"
    assert len(full_sel) == total, f"{len(full_sel):,} tiles != {total:,} expected"
    print(
        f"wrote {OUT}/slides/slides.parquet ({len(slides_sel)} rows) and "
        f"{OUT}/tiles/tiles.parquet ({len(full_sel):,} rows)"
    )


if __name__ == "__main__":
    main()
