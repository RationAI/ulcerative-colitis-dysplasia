#!/usr/bin/env python
"""Re-stamp the 7 inspect slides with ALL 9 annotation classes and build
per-class level-3 tile masks from the restamped table.

Why this exists: tiling_200 only carries the two dysplasia classes (tiling
was run with target_groups=[lowgrade_dysplasia, highgrade_dysplasia]), but the
EMPIA JSONs (preprocessing/annotations/, converted losslessly from the
dataset's per-tile label CSVs by scripts/make_huncrc_annotations.py) hold all
nine. To inspect the other seven classes we re-derive the per-tile coverage
fractions offline -- no re-tiling, no WSI pixels:

  * tiling level = level 2 (downsample 4.0), tile 224 px = 896 level-0 px
  * label polygons are axis-aligned squares on a regular 1024 level-0 px
    grid (zoom_1 pitch; all 7 slides are zoom_1 -- the only zoom_2 slide is
    077, not in the set)
  * a tile (896 < pitch) therefore overlaps at most 2x2 = 4 label squares,
    so coverage is a grid lookup, vectorized over the tile table

Outputs (all NEW dirs; nothing existing is touched):
  preprocessing/tiling_7annot/
      slides/slides.parquet          the 7 rows of tiling_200's slides table
      tiles/{stem}.parquet           the slide's tiling_200 tile rows + 7 new
                                     0..1 coverage columns (normal,
                                     adenocarcinoma, artifact, inflammation,
                                     suspicious_for_invasion, tumor_necrosis,
                                     resection_edge). Original columns kept
                                     as-is.
  preprocessing/tile_masks_allclasses/{class}/{stem}.tiff
      level-3 big-TIFFs (same canvas/geometry as the tile_masks and MLP
      prediction masks: tile 224 -> 112 px, stride 112 -> 56 px, 2x2-tile
      mean, mpp x2), one per class per slide, written immediately after
      each class is rasterized.

Memory: per-slide peak is the tile table (~25 MB) plus a float64 lattice of
~n_tiles cells (~1 MB max); the upscale streams through pyvips. No step
materializes the level-3 canvas -- safe on the 32 GB pod.

Usage:
    python scripts/restamp_annot_tiles.py --stamp
    python scripts/restamp_annot_tiles.py --masks
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
TILING = PROJECT / "preprocessing" / "tiling_200"
ANNOT = PROJECT / "preprocessing" / "annotations"
T7 = PROJECT / "preprocessing" / "tiling_7annot"
MASKS = PROJECT / "preprocessing" / "tile_masks_allclasses"

SLIDES = ["139", "106", "121", "133", "090", "112", "124"]
#: the seven classes tiling_200 does not carry
NEW_CLASSES = [
    "normal",
    "adenocarcinoma",
    "artifact",
    "inflammation",
    "suspicious_for_invasion",
    "tumor_necrosis",
    "resection_edge",
]
#: all nine for the mask step (the two dysplasia classes as reference layers)
ALL_CLASSES = NEW_CLASSES + ["lowgrade_dysplasia", "highgrade_dysplasia"]

#: level-3 mask canvas geometry (tile_masks convention)
TILE_EXTENT = 224
STRIDE = 112
MASK_STRIDE = STRIDE // 2

BIG = 1 << 20  #: key = i*BIG + j, coords/pitch < 1000


def _slide_rows() -> pd.DataFrame:
    s = pd.read_parquet(TILING / "slides" / "slides.parquet")
    s["stem"] = s["path"].map(lambda p: Path(p).stem)
    s = s[s["stem"].isin(SLIDES)]
    assert len(s) == len(SLIDES), f"expected {len(SLIDES)} slides, got {len(s)}"
    return s.sort_values("stem").reset_index(drop=True)


_TILE_CACHE: dict[str, pd.DataFrame] = {}


def _tiles_for(sid: str) -> pd.DataFrame:
    """One pass over the 18 shards (the full table is 4.3M rows; we keep the
    ~336k of the 7 slides), cached per slide."""
    if not _TILE_CACHE:
        keep = set(_slide_rows()["id"])
        for f in sorted((TILING / "tiles").glob("*.parquet")):
            t = pd.read_parquet(f)
            t = t[t["slide_id"].isin(keep)]
            if len(t):
                for s in t["slide_id"].unique():
                    _TILE_CACHE[s] = pd.concat(
                        [_TILE_CACHE.get(s), t[t["slide_id"] == s]],
                        ignore_index=True,
                    )
    return _TILE_CACHE[sid]


def _label_squares(stem: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """class -> (topleft_x, topleft_y) arrays in level-0 px + the measured
    pitch; asserts the squares sit on one regular grid."""
    items = json.loads((ANNOT / f"{stem}.json").read_text())["items"]
    by_class: dict[str, list[tuple[float, float, float]]] = {}
    for it in items:
        c = it["coordinates"]
        by_class.setdefault(it["name"], []).append(
            (c[0][0], c[0][1], c[1][0] - c[0][0])
        )
    sides = {s for v in by_class.values() for _, _, s in v}
    assert len(sides) == 1, f"{stem}: mixed square sides {sides}"
    pitch = int(round(sides.pop()))
    out = {
        cls: (
            np.array([p[0] for p in v], dtype=np.int64),
            np.array([p[1] for p in v], dtype=np.int64),
        )
        for cls, v in by_class.items()
    }
    return out, pitch


def _coverage(
    n: int,
    tx0: np.ndarray,
    ty0: np.ndarray,
    ext: int,
    pitch: int,
    squares: dict[str, tuple[np.ndarray, np.ndarray]],
    classes: list[str],
) -> dict[str, np.ndarray]:
    """Per-tile coverage fraction (0..1) per class -- grid lookup with a
    per-class origin offset (a slide's label grid can be shifted from the
    slide origin, e.g. 090) and gaps; at most 2x2 squares per tile
    (tile extent < pitch)."""
    tile_area = float(ext * ext)
    covs = {}
    for cls in classes:
        cov = np.zeros(n)
        sx, sy = squares.get(cls, (None, None))
        if sx is not None and len(sx):
            #: all squares of one class share a single origin mod pitch
            ox = int(sx[0] % pitch)
            oy = int(sy[0] % pitch)
            assert not ((sx - ox) % pitch).any(), f"{cls}: x off its own grid"
            assert not ((sy - oy) % pitch).any(), f"{cls}: y off its own grid"
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
        covs[cls] = np.minimum(cov / tile_area, 1.0)
    return covs


def stamp() -> None:
    (T7 / "slides").mkdir(parents=True, exist_ok=True)
    (T7 / "tiles").mkdir(parents=True, exist_ok=True)
    slides = _slide_rows()
    slides.drop(columns=["stem"]).to_parquet(T7 / "slides" / "slides.parquet")

    for row in slides.itertuples():
        stem, sid = row.stem, row.id
        t = _tiles_for(sid)
        ds = float(row.downsample)
        ext = int(round(float(row.tile_extent_x) * ds))
        assert abs(float(row.tile_extent_y) - row.tile_extent_x) < 1e-6
        tx0 = (t["x"].to_numpy() * ds).astype(np.int64)
        ty0 = (t["y"].to_numpy() * ds).astype(np.int64)
        squares, pitch = _label_squares(stem)
        covs = _coverage(
            len(t), tx0, ty0, ext, pitch, squares,
            NEW_CLASSES + ["lowgrade_dysplasia", "highgrade_dysplasia"],
        )
        #: save all nine re-derived coverages (incl. the two dysplasia classes) so
        #: the mask step runs every layer off the current annotations, not the
        #: (stale) tiling columns
        for cls in ALL_CLASSES:
            t[f"annot_{cls}"] = covs[cls]

        #: cross-check: re-derived dysplasia coverage vs the tiling columns
        for ref, mine in (
            ("lowgrade_dysplasia", covs["lowgrade_dysplasia"]),
            ("highgrade_dysplasia", covs["highgrade_dysplasia"]),
        ):
            diff = np.abs(t[ref].to_numpy() - mine)
            print(
                f"{stem}: {ref} re-derived vs tiling -- max diff {diff.max():.2e}, "
                f"mismatches>1e-9 {int((diff > 1e-9).sum())}/{len(t)}",
                flush=True,
            )

        t.to_parquet(T7 / "tiles" / f"{stem}.parquet")
        counts = {
            c: int((t[f"annot_{c}"] > 0).sum()) for c in NEW_CLASSES
        }
        print(f"{stem}: {len(t)} tiles, pitch {pitch}, " + ", ".join(
            f"{c} {n}" for c, n in counts.items()
        ), flush=True)
    print(f"stamp done -> {T7}")


def masks() -> None:
    import pyvips
    from ratiopath.masks import write_big_tiff
    from ratiopath.openslide import OpenSlide

    slides = _slide_rows()
    for row in slides.itertuples():
        stem, sid = row.stem, row.id
        t = pd.read_parquet(T7 / "tiles" / f"{stem}.parquet")
        with OpenSlide(row.path) as ws:
            w3, h3 = ws.level_dimensions[3]
            mpp_x, mpp_y = ws.slide_resolution(3)

        nrow, ncol = h3 // MASK_STRIDE + 1, w3 // MASK_STRIDE + 1
        kx = (t["x"].to_numpy() // STRIDE).astype(np.int64)
        ky = (t["y"].to_numpy() // STRIDE).astype(np.int64)
        assert len(np.unique(ky * BIG + kx)) == len(t), f"{stem}: duplicate lattice cells"

        for cls in ALL_CLASSES:
            #: every layer from the re-derived (current-annotation) coverage
            col = f"annot_{cls}"
            L = np.zeros((nrow, ncol))
            L[ky, kx] = t[col].to_numpy()
            #: count-based 2x2 mean (same as the prediction masks): a tile
            #: beside non-tissue keeps its own value, not a diluted one
            present = L > 0
            Sf = np.pad(np.where(present, L, 0.0), ((1, 0), (1, 0)))
            Sn = np.pad(present.astype(np.float64), ((1, 0), (1, 0)))
            num = (
                Sf[:-1, :-1] + Sf[1:, :-1] + Sf[:-1, 1:] + Sf[1:, 1:]
            )
            den = (
                Sn[:-1, :-1] + Sn[1:, :-1] + Sn[:-1, 1:] + Sn[1:, 1:]
            )
            A = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            im = pyvips.Image.new_from_array((A * 255).astype(np.uint8))
            im = im.resize(MASK_STRIDE, kernel="nearest")
            im = im.crop(0, 0, w3, h3)
            out = MASKS / cls / f"{stem}.tiff"
            out.parent.mkdir(parents=True, exist_ok=True)
            write_big_tiff(im, out, mpp_x, mpp_y)
            print(
                f"{stem}/{cls}: {int((L > 0).sum())} tiles, max {L.max():.3f}, "
                f"{out.stat().st_size / 1e6:.1f} MB",
                flush=True,
            )
        print(f"=== {stem} done (9/9 classes)", flush=True)
    print("masks done ->", MASKS, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stamp", action="store_true",
                    help="re-stamp all 9 classes -> tiling_7annot parquets")
    ap.add_argument("--masks", action="store_true",
                    help="9 per-class level-3 tiffs per slide -> tile_masks_allclasses")
    args = ap.parse_args()
    if args.stamp:
        stamp()
    if args.masks:
        masks()
