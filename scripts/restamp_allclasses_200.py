#!/usr/bin/env python
"""Re-stamp ALL 200 hunCRC slides with all 9 annotation classes.

Cohort-wide version of scripts/restamp_annot_tiles.py (which did the 7
inspect slides). Same trick: tiling_200 only carries the two dysplasia
classes, but the EMPIA JSONs (preprocessing/annotations/, converted
losslessly from the dataset's per-tile label CSVs by
scripts/make_huncrc_annotations.py) hold all nine, as axis-aligned squares
on a regular level-0 grid (pitch 1024 px for the zoom_1 slides, 2048 for
the one zoom_2 slide -- verified across all 200 JSONs: 408,915 squares,
vocabulary = exactly the 9 classes, no others). A 224-px tile at level 2
(896 level-0 px) overlaps at most 2x2 squares, so per-tile coverage is a
vectorized grid lookup -- no re-tiling, no WSI pixels, no shapely.

Why: the new training-label policy (doc.md, "Label policy v2") needs the
per-tile coverage of all nine classes for every tile, plus a definitive
per-slide cross-check of re-derived vs tiling_200 dysplasia columns (the
132/200 tiling-vs-annotations provenance question, quantified cohort-wide).

Output (NEW dir; nothing existing is touched):
    preprocessing/tiling_200_allclasses/tiles/{stem}.parquet
        the slide's tiling_200 tile rows + 9 `annot_{class}` coverage
        columns (0..1), original columns kept as-is.

Memory: one shard of tiling_200 tiles at a time (~240k rows, <100 MB with
usecols) plus the per-slide accumulator -- safe on the 32 GB pod.
Resumable: slides whose output parquet exists are skipped.

Usage:
    .venv/bin/python scripts/restamp_allclasses_200.py
    .venv/bin/python scripts/restamp_allclasses_200.py --slides 001 011 077
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
TILING = PROJECT / "preprocessing" / "tiling_200"
ANNOT = PROJECT / "preprocessing" / "annotations"
OUT = PROJECT / "preprocessing" / "tiling_200_allclasses" / "tiles"

ALL_CLASSES = [
    "normal",
    "adenocarcinoma",
    "artifact",
    "inflammation",
    "suspicious_for_invasion",
    "tumor_necrosis",
    "resection_edge",
    "lowgrade_dysplasia",
    "highgrade_dysplasia",
]

BIG = 1 << 20  #: key = i*BIG + j, coords/pitch < 1000 so no key collision

#: tile columns to keep (drops nothing semantic; the table is small anyway)
TILE_COLS = [
    "slide_id", "x", "y", "tissue", "annotation", "epithelium", "blur",
    "artifacts", "clarity", "fold", "split",
    "lowgrade_dysplasia", "highgrade_dysplasia",
]


def _slides() -> pd.DataFrame:
    s = pd.read_parquet(TILING / "slides" / "slides.parquet")
    s["stem"] = s["path"].map(lambda p: Path(p).stem)
    return s.sort_values("stem").reset_index(drop=True)


def _label_squares(stem: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int] | None:
    """class -> (topleft_x, topleft_y) arrays in level-0 px, plus the one
    square side (pitch) on the slide. None when the slide has no annotation
    items at all."""
    items = json.loads((ANNOT / f"{stem}.json").read_text())["items"]
    if not items:
        return None
    by_class: dict[str, list[tuple[float, float, float]]] = {}
    for it in items:
        c = it["coordinates"]
        by_class.setdefault(it["name"], []).append(
            (c[0][0], c[0][1], c[1][0] - c[0][0])
        )
    sides = {s for v in by_class.values() for _, _, s in v}
    assert len(sides) == 1, f"{stem}: mixed square sides {sides}"
    pitch = int(round(sides.pop()))
    return (
        {
            cls: (
                np.array([p[0] for p in v], dtype=np.int64),
                np.array([p[1] for p in v], dtype=np.int64),
            )
            for cls, v in by_class.items()
        },
        pitch,
    )


def _coverage(
    n: int,
    tx0: np.ndarray,
    ty0: np.ndarray,
    ext: int,
    pitch: int,
    squares: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> dict[str, np.ndarray]:
    """Per-tile coverage fraction (0..1) per class -- grid lookup with a
    per-class origin offset (a slide's label grid can be shifted from the
    slide origin, e.g. 090) and gaps. Offsets per axis = ext//pitch + 2,
    a safe bound for any pitch (2 for this cohort's 1024/2048)."""
    tile_area = float(ext * ext)
    covs = {cls: np.zeros(n) for cls in ALL_CLASSES}
    if squares is None:
        return covs
    na = ext // pitch + 2
    for cls in ALL_CLASSES:
        cov = covs[cls]
        sx, sy = squares.get(cls, (None, None))
        if sx is None or not len(sx):
            continue
        #: all squares of one class share a single origin mod pitch
        ox = int(sx[0] % pitch)
        oy = int(sy[0] % pitch)
        assert not ((sx - ox) % pitch).any(), f"{cls}: x off its own grid"
        assert not ((sy - oy) % pitch).any(), f"{cls}: y off its own grid"
        occ = np.unique(((sx - ox) // pitch) * BIG + ((sy - oy) // pitch))
        i0 = (tx0 - ox) // pitch
        j0 = (ty0 - oy) // pitch
        for a in range(na):
            i = i0 + a
            for b in range(na):
                j = j0 + b
                m = np.isin(i * BIG + j, occ)
                if not m.any():
                    continue
                oxl = np.clip(
                    np.minimum(tx0 + ext, ox + (i + 1) * pitch)
                    - np.maximum(tx0, ox + i * pitch),
                    0, None,
                )
                oyl = np.clip(
                    np.minimum(ty0 + ext, oy + (j + 1) * pitch)
                    - np.maximum(ty0, oy + j * pitch),
                    0, None,
                )
                cov += np.where(m, oxl * oyl, 0.0)
        covs[cls] = np.minimum(cov / tile_area, 1.0)
    return covs


def stamp(subset: list[str] | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    slides = _slides()
    if subset:
        slides = slides[slides["stem"].isin(subset)].reset_index(drop=True)
    pending = slides[~slides["stem"].map(lambda s: (OUT / f"{s}.parquet").exists())]
    print(f"{len(slides)} slides, {len(pending)} to do, {len(slides) - len(pending)} cached", flush=True)

    #: accumulate tiles per slide across shards (a slide may span shards)
    acc: dict[str, list[pd.DataFrame]] = {sid: [] for sid in pending["id"]}
    t0 = time.time()
    for f in sorted((TILING / "tiles").glob("*.parquet")):
        sh = pd.read_parquet(f, columns=TILE_COLS)
        sh = sh[sh["slide_id"].isin(set(acc))]
        if not len(sh):
            continue
        for sid, part in sh.groupby("slide_id", sort=False):
            acc[sid].append(part)
        sh = None

    n_done = 0
    for row in pending.itertuples():
        stem, sid = row.stem, row.id
        t = pd.concat(acc[sid], ignore_index=True)
        acc[sid] = None  #: free it
        ds = float(row.downsample)
        ext = int(round(float(row.tile_extent_x) * ds))
        assert abs(float(row.tile_extent_y) - row.tile_extent_x) < 1e-6
        tx0 = (t["x"].to_numpy() * ds).astype(np.int64)
        ty0 = (t["y"].to_numpy() * ds).astype(np.int64)
        res = _label_squares(stem)
        squares, pitch = res if res is not None else (None, 0)
        covs = _coverage(len(t), tx0, ty0, ext, pitch, squares)
        for cls in ALL_CLASSES:
            t[f"annot_{cls}"] = covs[cls]

        #: cross-check re-derived dysplasia vs the tiling_200 columns
        diffs = {
            ref: np.abs(t[ref].to_numpy() - covs[ref])
            for ref in ("lowgrade_dysplasia", "highgrade_dysplasia")
        }
        bad = sum(int((d > 1e-9).sum()) for d in diffs.values())
        t.to_parquet(OUT / f"{stem}.parquet")
        n_done += 1
        counts = {c: int((t[f"annot_{c}"] > 0).sum()) for c in ALL_CLASSES}
        print(
            f"[{n_done}/{len(pending)}] {stem}: {len(t)} tiles, pitch {pitch}, "
            f"mismatch>1e-9 {bad}/{len(t)} | " + ", ".join(
                f"{c.split('_')[0]} {n}" for c, n in counts.items() if n
            ),
            flush=True,
        )
    print(f"stamp done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slides", nargs="*", default=None,
                    help="stamp only these stems (default: all 200)")
    args = ap.parse_args()
    stamp(args.slides)
