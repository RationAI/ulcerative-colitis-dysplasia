#!/usr/bin/env python
"""Tile only the hunCRC slides missing from preprocessing/tiling (the 14-slide QC gap).

Reuses preprocessing/tiling.py's tiling() as-is with the same geometry as the
186-slide job (224 px @ 0.52 mpp, stride 112, tissue >= 0.5 -- the proven pilot
values from scripts/preprocessing/tiling.py), so the output carries the same
columns and semantics as the existing tiling. Everything is read from the
share directly (no MLflow artifacts needed):

  QC            preprocessing/quality_control         (complete 200-slide QC)
  tissue        preprocessing/tissue_masks            (200 masks)
  epithelium    preprocessing/epithelium_masks/raw    (200 real epi masks)
  annotations   preprocessing/annotations             (200 EMPIA JSONs)
  split/fold    preprocessing/dataset_split.csv

Two post-processing steps make the columns match the current tiling exactly:

  * `split` is stamped the same way scripts/add_split_column.py does
    (slides joined on stem, tiles on slide_id -> slides.id; here the file name
    IS the stem, so the lookup is direct).
  * `epithelium` is OVERWRITTEN with the same full-tile binary fraction that
    scripts/compute_epithelium_column.py produced for the 186 (fraction of the
    full 224x224 tile with epi probability >= 0.5, on the level-2 mask page).
    tiling.py's own epi overlap is a half-tile probability MEAN -- a different
    metric -- so keeping it would make the 14 inconsistent with the 186.

Output: preprocessing/tiling_missing14/{slides,tiles}/{stem}.parquet -- ONE FILE
PER SLIDE. That is what lets the 14 slides run as 14 concurrent cluster jobs
(scripts/preprocessing/tiling_missing14.py): each job tiles its one slide and
writes only its own files, so no two jobs ever touch the same file. A slide is
~1.6M candidate tiles -> ~12k-140k kept rows of small columns, so
materializing per slide is trivial.

Nothing existing is touched. Merge afterwards with scripts/merge_tiling_missing14.py.

Usage (from the notebook pod; .venv/bin/python, never `uv run` -- Ray workers
hang under it):
    .venv/bin/python scripts/tiling_missing14.py                  # all missing slides
    .venv/bin/python scripts/tiling_missing14.py --slides 011     # one slide (job mode)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
#: sys.path[0] is scripts/, where `preprocessing/` holds the JOB SUBMITTERS --
#: importing scripts/preprocessing/tiling.py executes submit_job. The repo root
#: (the real `preprocessing` package) must win.
sys.path.insert(0, str(REPO_ROOT))

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
QC = PROJECT / "preprocessing" / "quality_control"
TISSUE = PROJECT / "preprocessing" / "tissue_masks"
EPI = PROJECT / "preprocessing" / "epithelium_masks" / "raw"
ANNOT = PROJECT / "preprocessing" / "annotations"
SPLIT_CSV = PROJECT / "preprocessing" / "dataset_split.csv"
TILING = PROJECT / "preprocessing" / "tiling"
OUT = PROJECT / "preprocessing" / "tiling_missing14"

#: Same geometry as the 186-slide job -- do not change without re-checking.
MPP = 0.52
TILE_EXTENT = 224
STRIDE = 112
TISSUE_THRESHOLD = 0.5
TARGET_GROUPS = ["lowgrade_dysplasia", "highgrade_dysplasia"]
#: 0.5 probability on the 0-255 epi heatmap -- same as compute_epithelium_column.
EPI_THRESHOLD = 128
#: Mask rows per read in the epi pass; bounds memory (~140 MB per block).
BLOCK_ROWS = 1024


def load_split() -> pd.DataFrame:
    df = pd.read_csv(
        SPLIT_CSV, index_col="slide_id", dtype={"slide_id": str, "case_id": str}
    )
    df.index = df.index.str.zfill(3)
    return df


def missing_stems(split_df: pd.DataFrame) -> list[str]:
    slides = pd.read_parquet(TILING / "slides" / "slides.parquet", columns=["path"])
    tiled = {Path(p).stem for p in slides["path"]}
    return sorted(set(split_df.index) - tiled)


def preflight(stems: list[str]) -> None:
    missing = [
        str(f)
        for s in stems
        for f in (
            TISSUE / f"{s}.tiff",
            EPI / f"{s}.tiff",
            ANNOT / f"{s}.json",
            QC / "blur_per_pixel" / f"{s}.tiff",
            QC / "artifacts_per_pixel" / f"{s}.tiff",
        )
        if not f.exists()
    ]
    if missing:
        sys.exit("missing inputs:\n" + "\n".join(missing))
    qc = pd.read_csv(QC / "qc_metrics.csv")
    qc_idx = {str(int(s)).zfill(3) for s in qc["slide_name"].astype(str) if s.isdigit()}
    no_qc = [s for s in stems if s not in qc_idx]
    if no_qc:
        sys.exit(f"not in qc_metrics.csv: {no_qc}")


def run_tiling(df: pd.DataFrame) -> None:
    import ray

    from preprocessing.tiling import tiling

    (OUT / "slides").mkdir(parents=True, exist_ok=True)
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)

    ray.init(runtime_env={"excludes": [".git", ".venv"]})
    try:
        ds_slides, ds_tiles = tiling(
            df,
            qc_folder=QC,
            tissue_folder=TISSUE,
            epithelium_folder=EPI,
            annot_folder=ANNOT,
            tile_extent=TILE_EXTENT,
            stride=STRIDE,
            mpp=MPP,
            tissue_threshold=TISSUE_THRESHOLD,
            target_groups=TARGET_GROUPS,
        )
        #: One parquet per slide -- see the module docstring for why. Materializing
        #: is cheap: a slide keeps ~12k-140k rows of ~13 small columns.
        slides_pd = ds_slides.to_pandas()
        tiles_pd = ds_tiles.to_pandas()
        slides_pd = slides_pd.assign(
            stem=slides_pd["path"].map(lambda p: Path(p).stem)
        )
        stem_of = dict(zip(slides_pd["id"], slides_pd["stem"]))
        for stem, grp in slides_pd.groupby("stem", sort=True):
            grp.drop(columns="stem").to_parquet(
                OUT / "slides" / f"{stem}.parquet", index=False
            )
        for sid, grp in tiles_pd.groupby("slide_id", sort=False):
            grp.to_parquet(OUT / "tiles" / f"{stem_of[sid]}.parquet", index=False)
    finally:
        ray.shutdown()


def stamp_split(split_df: pd.DataFrame) -> None:
    #: File name == slide stem, so the split lookup is direct on both file kinds.
    for f in sorted((OUT / "slides").glob("*.parquet")):
        s = pd.read_parquet(f)
        if f.stem not in set(split_df.index):
            sys.exit(f"{f.name}: not in dataset_split.csv")
        s["split"] = split_df.loc[f.stem, "split"]
        s.to_parquet(f, index=False)
    for f in sorted((OUT / "tiles").glob("*.parquet")):
        t = pd.read_parquet(f)
        if f.stem not in set(split_df.index):
            sys.exit(f"{f.name}: not in dataset_split.csv")
        t["split"] = split_df.loc[f.stem, "split"]
        t.to_parquet(f, index=False)
        print(f"split -> {f.name}: {t['split'].value_counts().to_dict()}")


def epi_fraction(stem: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-tile fraction of the FULL tile with epi probability >= EPI_THRESHOLD,
    streaming the mask's level-2 top page in row blocks (same algorithm as
    scripts/compute_epithelium_column.py). Returns fractions in (x, y) order."""
    import zarr
    from tifffile import imread

    xs, col = np.unique(x, return_inverse=True)
    store = imread(str(EPI / f"{stem}.tiff"), aszarr=True)
    try:
        page = zarr.open(store, mode="r")["0"]
        H, W = page.shape
        bad = int(((x + TILE_EXTENT > W) | (y + TILE_EXTENT > H)).sum())
        if bad:
            sys.exit(f"{stem}: {bad} tiles extend past the level-2 extent {W}x{H}")
        #: Column-window sums: C[r, j] = epi pixels in row r, cols [xs[j], xs[j]+TILE).
        C = np.zeros((H, len(xs)), dtype=np.uint16)
        for r0 in range(0, H, BLOCK_ROWS):
            block = np.asarray(page[r0:r0 + BLOCK_ROWS]) >= EPI_THRESHOLD
            for j, xv in enumerate(xs):
                C[r0:r0 + len(block), j] = (
                    block[:, xv:xv + TILE_EXTENT].sum(axis=1, dtype=np.uint16)
                )
    finally:
        store.close()
    Cp = np.zeros((H + 1, len(xs)), dtype=np.uint32)
    np.cumsum(C, axis=0, dtype=np.uint32, out=Cp[1:])
    del C
    frac = (Cp[y + TILE_EXTENT, col] - Cp[y, col]) / (TILE_EXTENT * TILE_EXTENT)
    assert np.all((frac >= 0.0) & (frac <= 1.0))
    return frac


def recompute_epithelium() -> None:
    #: Tiles are per-slide files, so each file is one epi_fraction call.
    for f in sorted((OUT / "tiles").glob("*.parquet")):
        t = pd.read_parquet(f)
        t["epithelium"] = epi_fraction(f.stem, t["x"].to_numpy(), t["y"].to_numpy())
        print(f"epi {f.stem}: {len(t)} tiles, mean {t['epithelium'].mean():.3f}",
              flush=True)
        t.to_parquet(f, index=False)


def report() -> None:
    import pyarrow.parquet as pq

    #: Column check via metadata only -- no need to load the data pages.
    cur_cols = pq.read_schema(sorted(TILING.glob("tiles/*.parquet"))[0]).names
    for f in sorted((OUT / "tiles").glob("*.parquet")):
        new_cols = pq.read_schema(f).names
        if new_cols != cur_cols:
            sys.exit(f"{f.name} columns differ:\n{new_cols}\nvs\n{cur_cols}")
    slides = pd.concat(
        (pd.read_parquet(f) for f in sorted((OUT / "slides").glob("*.parquet"))),
        ignore_index=True,
    )
    slides = slides.assign(stem=slides["path"].map(lambda p: Path(p).stem))
    print(f"\n{len(slides)} slides -> {OUT}")
    print(slides[["stem", "split", "fold"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--slides", default=None,
        help="comma list of stems; default = everything in dataset_split.csv "
             "that is not in preprocessing/tiling",
    )
    args = ap.parse_args()

    split_df = load_split()
    if args.slides:
        stems = sorted({s.zfill(3) for s in args.slides.split(",") if s.strip()})
    else:
        stems = missing_stems(split_df)
    if not stems:
        sys.exit("nothing to do -- every split slide is already tiled")
    print(f"tiling {len(stems)} slides: {', '.join(stems)}")
    unknown = [s for s in stems if s not in set(split_df.index)]
    if unknown:
        sys.exit(f"not in dataset_split.csv: {unknown}")
    preflight(stems)

    run_tiling(split_df.loc[stems])
    stamp_split(split_df)
    recompute_epithelium()
    report()
    print("\nnext: .venv/bin/python scripts/merge_tiling_missing14.py")


if __name__ == "__main__":
    main()
