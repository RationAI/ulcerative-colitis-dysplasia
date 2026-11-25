"""Fill the existing (all-None) `fold` column in tiling_186_merged.

Companion to scripts/add_split_column.py: the split was written to the
parquets but the pre-existing `fold` column was left untouched. This fills it
from dataset_split.csv (the train rows carry 0-4 from the 5-fold GroupKFold;
test rows stay None -- the same convention as the IBD pipeline, where only
train tiles carry a fold).

    python scripts/fill_fold_column.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


DEFAULT_SPLIT_CSV = "/mnt/projects/huncrc_dysplasia/dataset_split.csv"
DEFAULT_TILING_DIR = "/mnt/projects/huncrc_dysplasia/tiling_186_merged"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    ap.add_argument("--tiling-dir", default=DEFAULT_TILING_DIR)
    args = ap.parse_args()

    tiling_dir = Path(args.tiling_dir)
    #: index_col + dtype str: the stems are zero-padded, read_csv infers int64
    #: otherwise and the slide-stem join fails for every row.
    split_df = pd.read_csv(
        args.split_csv, index_col="slide_id", dtype={"slide_id": str}
    )

    slides = pd.read_parquet(tiling_dir / "slides" / "slides.parquet")
    stems = slides["path"].map(lambda p: Path(p).stem)
    unknown = sorted(set(stems) - set(split_df.index))
    if unknown:
        sys.exit(f"{len(unknown)} tiled slides missing from split csv: {unknown[:10]}")

    slides["fold"] = stems.map(lambda s: split_df.loc[s, "fold"])
    slides.to_parquet(tiling_dir / "slides" / "slides.parquet", index=False)
    print(f"slides.parquet: {slides['fold'].value_counts(dropna=False).to_dict()}")

    slide_fold = slides.set_index("id")["fold"]
    for f in sorted(glob.glob(str(tiling_dir / "tiles" / "*.parquet"))):
        t = pd.read_parquet(f)
        t["fold"] = t["slide_id"].map(slide_fold)
        tmp = f + ".tmp"
        t.to_parquet(tmp, index=False)
        Path(tmp).replace(f)
        print(
            f"  {Path(f).name}: {len(t)} tiles, "
            f"{t['fold'].value_counts(dropna=False).to_dict()}"
        )

    print("done")


if __name__ == "__main__":
    main()
