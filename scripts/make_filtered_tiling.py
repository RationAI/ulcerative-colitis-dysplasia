"""Cluster-filter the hunCRC tiling: keep only the annotated x-gap clusters.

For each slide that carries at least one annotated tile, tiles are clustered
by x-gap (a gap > 5x the MEDIAN gap between distinct x starts a new cluster --
the same rule as the multi-cluster analysis and the old IBD dataset filter).
Every cluster carrying annotations is kept whole; all other clusters on that
slide are dropped. Slides without any annotation are kept untouched
(true-negative slides).

Row values are never modified -- rows are only dropped. Column schema is
preserved.

Writes a NEW tiling directory with the same layout (slides/slides.parquet +
tiles/tiles_000{i}.parquet). Refuses to run if the output directory exists
and is non-empty, so the original tiling can never be overwritten. A
per-slide filter_report.csv is written into the output directory.

Usage (from repo root):
    .venv/bin/python scripts/make_filtered_tiling.py \
        [--in /mnt/projects/huncrc_dysplasia/tiling_186_merged] \
        [--out /mnt/projects/huncrc_dysplasia/tiling_186_merged_filt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NSHARDS = 4
#: A gap > GAP_FACTOR x the median gap between distinct x starts a new cluster.
GAP_FACTOR = 5.0


def cluster_ids(df: pd.DataFrame) -> pd.Series:
    """Per-row x-gap cluster id (same heuristic as the old IBD filter)."""
    unique_x = np.sort(df["x"].unique())
    diffs = np.diff(unique_x)
    threshold = np.median(diffs) * GAP_FACTOR if diffs.size else 0.0
    clusters = np.concatenate([[0], (diffs > threshold).cumsum()])
    return df["x"].map(dict(zip(unique_x.tolist(), clusters.tolist())))


def filter_slide(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep the annotation-bearing clusters of one slide; drop the rest."""
    n_clusters = int(cluster_ids(df).nunique()) if len(df) else 0
    if int(df["annotation"].sum()) == 0:
        # Zero-annotation slide: true-negative, keep whole.
        return df, {
            "n_clusters": n_clusters,
            "n_annotated_clusters": 0,
            "n_before": len(df),
            "n_kept": len(df),
            "n_dropped": 0,
        }
    cl = cluster_ids(df)
    per_cluster = cl.groupby(cl).agg(annotation=("annotation", "sum"))
    valid = set(per_cluster.index[per_cluster["annotation"] > 0])
    kept = df[cl.isin(valid)]
    return kept, {
        "n_clusters": n_clusters,
        "n_annotated_clusters": len(valid),
        "n_before": len(df),
        "n_kept": len(kept),
        "n_dropped": len(df) - len(kept),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--in", dest="in_dir",
        default="/mnt/projects/huncrc_dysplasia/tiling_186_merged",
    )
    ap.add_argument(
        "--out", dest="out_dir",
        default="/mnt/projects/huncrc_dysplasia/tiling_186_merged_filt",
    )
    args = ap.parse_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    if in_dir.resolve() == out_dir.resolve():
        raise SystemExit("--out must differ from --in")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty -- refusing to overwrite")

    tiles = pd.concat(
        [pd.read_parquet(f) for f in sorted((in_dir / "tiles").glob("*.parquet"))],
        ignore_index=True,
    )
    n0 = len(tiles)
    pos0 = int(
        ((tiles["lowgrade_dysplasia"] > 0) | (tiles["highgrade_dysplasia"] > 0)).sum()
    )
    print(f"input: {n0} tiles ({pos0} dysplasia) from {in_dir}")

    report_rows, kept_parts = [], []
    for sid, df in tiles.groupby("slide_id", sort=False):
        kept, info = filter_slide(df)
        kept_parts.append(kept)
        info["slide_id"] = str(sid)
        report_rows.append(info)
    out = pd.concat(kept_parts, ignore_index=True)
    report = pd.DataFrame(report_rows)

    # -- write the new tiling dir ------------------------------------------
    (out_dir / "slides").mkdir(parents=True)
    (out_dir / "tiles").mkdir()
    pd.read_parquet(in_dir / "slides" / "slides.parquet").to_parquet(
        out_dir / "slides" / "slides.parquet", index=False
    )
    by_slide = {sid: out[out["slide_id"] == sid] for sid in sorted(out["slide_id"].unique())}
    ids = list(by_slide)
    for i in range(NSHARDS):
        shard = pd.concat([by_slide[s] for s in ids[i::NSHARDS]], ignore_index=True)
        shard.to_parquet(out_dir / "tiles" / f"tiles_{i:04d}.parquet", index=False)
    report.to_csv(out_dir / "filter_report.csv", index=False)

    # -- report --------------------------------------------------------------
    n1 = len(out)
    pos1 = int(
        ((out["lowgrade_dysplasia"] > 0) | (out["highgrade_dysplasia"] > 0)).sum()
    )
    print(
        f"output: {n1} tiles ({pos1} dysplasia) -> {out_dir} "
        f"(dropped {n0 - n1}, {100 * (n0 - n1) / n0:.1f}%)"
    )
    for split in ("train", "test_preliminary", "test_final"):
        b = int((tiles["split"] == split).sum())
        a = int((out["split"] == split).sum())
        print(f"  {split}: {b} -> {a} tiles")
    aff = report[report["n_dropped"] > 0]
    print(f"slides touched: {len(aff)} (of {len(report)})")
    for _, r in aff.iterrows():
        print(
            f"  {r['slide_id'][:8]}  clusters {r['n_clusters']} "
            f"(annotated {int(r['n_annotated_clusters'])})  "
            f"{r['n_before']} -> {r['n_kept']} tiles (-{r['n_dropped']})"
        )


if __name__ == "__main__":
    main()
