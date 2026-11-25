#!/usr/bin/env python
"""Cross-check tiling_200's dysplasia coverage columns against a fresh
re-derivation from the current EMPIA JSONs, for ALL 200 hunCRC slides.

Why: the Oct 29 tiling job read its EMPIA JSONs from
${project_path}/annotations -- a directory that no longer exists (the JSONs
now live in preprocessing/annotations, a copy dated Oct 1). On 7 inspect
slides, 3 (090/106/133) show LGD column values the current JSONs cannot
reproduce, while the other 4 match to machine precision -- i.e. the job read
a different JSON generation for at least those slides. This script says
whether the disagreement is cohort-wide or isolated.

Read-only: touches nothing, prints one line per disagreeing slide + a
summary. Peak RAM ~ a couple hundred MB (one shard at a time).

Usage:
    python scripts/check_tiling_vs_json.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
TILING = PROJECT / "preprocessing" / "tiling_200"
ANNOT = PROJECT / "preprocessing" / "annotations"

BIG = 1 << 20  #: cell key = i*BIG + j


def _coverage(n, tx0, ty0, ext, pitch, sx, sy):
    """Per-tile 0..1 coverage of the squares -- same grid lookup as
    restamp_annot_tiles.py (per-class origin offset + gaps)."""
    cov = np.zeros(n)
    if len(sx):
        ox = int(sx[0] % pitch)
        oy = int(sy[0] % pitch)
        assert not ((sx - ox) % pitch).any(), "x off its own grid"
        assert not ((sy - oy) % pitch).any(), "y off its own grid"
        occ = np.unique(((sx - ox) // pitch) * BIG + ((sy - oy) // pitch))
        i0 = (tx0 - ox) // pitch
        j0 = (ty0 - oy) // pitch
        for a in (0, 1):
            i = i0 + a
            for b in (0, 1):
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
    return np.minimum(cov / float(ext * ext), 1.0)


def _squares(stem: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    items = json.loads((ANNOT / f"{stem}.json").read_text())["items"]
    by_class: dict[str, list[tuple[float, float, float]]] = {}
    for it in items:
        c = it["coordinates"]
        by_class.setdefault(it["name"], []).append(
            (c[0][0], c[0][1], c[1][0] - c[0][0])
        )
    sides = {s for v in by_class.values() for _, _, s in v}
    assert len(sides) == 1, f"{stem}: mixed square sides {sides}"
    #: one label square per dataset tile, side == grid pitch (1024 zoom_1 /
    #: 2048 zoom_2) -- so the pitch IS the side, no measurement needed
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


def main() -> None:
    slides = pd.read_parquet(TILING / "slides" / "slides.parquet")
    slides["stem"] = slides["path"].map(lambda p: Path(p).stem)
    stem2ds = dict(zip(slides["stem"], slides["downsample"]))

    cache: dict[
        str, tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]
    ] = {}

    def squares_for(stem: str):
        if stem not in cache:
            cache[stem] = _squares(stem)
        return cache[stem]

    bad: dict[tuple[str, str], list[int]] = {}  # (stem, cls) -> [nmm, n]
    errs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def check_slide(stem: str, g: pd.DataFrame, ds: float) -> None:
        tx0 = (g["x"].to_numpy() * ds).astype(np.int64)
        ty0 = (g["y"].to_numpy() * ds).astype(np.int64)
        sq, pitch = squares_for(stem)
        ext = int(round(224 * ds))
        for cls in ("lowgrade_dysplasia", "highgrade_dysplasia"):
            sx, sy = sq.get(cls, (np.array([], dtype=np.int64),
                                  np.array([], dtype=np.int64)))
            mine = _coverage(len(g), tx0, ty0, ext, pitch, sx, sy)
            diff = np.abs(g[cls].to_numpy() - mine)
            nmm = int((diff > 1e-9).sum())
            #: a slide can span two 50k-row shards -- accumulate per slide
            acc = bad.setdefault((stem, cls), [0, 0, 0.0])
            acc[1] += len(g)  #: denominator always, for the full tile count
            if nmm:
                acc[0] += nmm
                acc[2] = max(acc[2], float(diff.max()))

    for f in sorted((TILING / "tiles").glob("*.parquet")):
        t = pd.read_parquet(
            f, columns=["slide_id", "x", "y",
                        "lowgrade_dysplasia", "highgrade_dysplasia"]
        )
        for sid, g in t.groupby("slide_id"):
            stem = str(slides.loc[slides["id"] == sid, "stem"].iloc[0])
            try:
                check_slide(stem, g, float(stem2ds[stem]))
            except Exception as e:  # keep sweeping; report at the end
                errs.append((stem, f"{type(e).__name__}: {e}"))
            seen.add(stem)

    bad = {k: v for k, v in bad.items() if v[0]}
    print(f"slides checked: {len(seen)}")
    print(f"slides with any mismatch: {len(set(s for s, _ in bad))}")
    for (stem, cls), (nmm, n, mx) in sorted(bad.items()):
        print(f"  {stem}: {cls} {nmm}/{n} tiles, max diff {mx:.3f}")
    if not bad:
        print("  (none -- current JSONs reproduce tiling_200 exactly)")
    for stem, err in sorted(set(errs)):
        print(f"  {stem}: SKIPPED -- {err}")
    if bad:
        out = Path("/tmp/tiling_vs_json_mismatches.csv")
        pd.DataFrame(
            [(s, c, v[0], v[1], v[2]) for (s, c), v in sorted(bad.items())],
            columns=["stem", "class", "mismatch_tiles", "total_tiles",
                     "max_abs_diff"],
        ).to_csv(out, index=False)
        print(f"full table -> {out}")


if __name__ == "__main__":
    main()
