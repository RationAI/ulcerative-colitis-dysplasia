r"""Build the labelled float32 feature cache (see ml/data/huncrc_features.py).

All paths come from arguments; nothing is hard-coded::

    python scripts/huncrc/build_feature_cache.py \
        --tiling <preprocessing>/tiling_200 \
        --slide-csv <TCIA_HunCRC>/slide_level_annotations.csv \
        --embeddings <preprocessing>/embeddings/prov-gigapath \
        --cache-root <local scratch> --splits train [--min-coverage 0.0]

The cache directory name carries min_coverage, the splits, the label-rule
fingerprint and the cache version. test_final is refused by the loader.
Prints peak RSS at the end.
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import pandas as pd

from ml.data.huncrc_features import build_features, load_features
from ml.data.huncrc_labels import LabelConfig, load_slide_table


def slide_table(tiling: Path, slide_csv: Path) -> pd.DataFrame:
    return load_slide_table(
        pd.read_parquet(tiling / "slides" / "slides.parquet"),
        pd.read_csv(slide_csv, dtype=str),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tiling", type=Path, required=True, help="dir with slides/slides.parquet"
    )
    p.add_argument(
        "--slide-csv", type=Path, required=True, help="slide_level_annotations.csv"
    )
    p.add_argument(
        "--embeddings", type=Path, required=True, help="dir of <slide>.parquet"
    )
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--min-coverage", type=float, default=0.0)
    p.add_argument(
        "--mode",
        choices=["labelled", "all"],
        default="labelled",
        help="labelled = label 0/1 tiles (training); all = every tile (evaluation)",
    )
    a = p.parse_args()
    t0 = time.time()
    config = LabelConfig(min_coverage=a.min_coverage)
    out = build_features(
        a.embeddings,
        slide_table(a.tiling, a.slide_csv),
        a.splits,
        config,
        a.cache_root,
        mode=a.mode,
    )
    fs = load_features(out, expected_config=config, mode=a.mode)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    size = fs.features.nbytes / 2**30
    print(
        f"{out}\n n={len(fs)} pos={fs.meta['n_pos']} neg={fs.meta['n_neg']} "
        f"features {fs.features.dtype} {fs.features.shape} = {size:.2f} GiB; "
        f"peak RSS {rss:.2f} GiB; {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
