#!/usr/bin/env python
"""Build the v2 training tile table (label policy v2, doc.md section 7).

Joins the restamped 9-class coverage (preprocessing/tiling_200_allclasses,
from scripts/restamp_allclasses_200.py) with the slide-level diagnosis
(slide_level_annotations.csv on the DATA share) and the QC columns already
carried by the tiling_200 tiles (per-tile blur / artifacts).

Policy (decided 2026-11-05, Adam Kukučka) -- first match wins:
  1. any LGD/HGD coverage                          -> POSITIVE (weight 1)
  2. QC blur  > --blur-threshold                   -> excluded:blur
  3. QC artifacts > --artifact-threshold           -> excluded:qc_artifact
  4. by slide category:
       negative / non_neoplastic_lesion:
         no neoplastic coverage on the tile        -> negative (weight 1)
           (unannotated tiles OK -- the slide diagnosis guarantees them;
           artifact/resection_edge-only tiles are the optional robust set,
           kept because step 3 already proved our QC does not flag them)
         non_neoplastic slide with a serrated / adenomatous polyp type
                                                   -> excluded:polyp_type
       adenoma / CRC:
         annotated, ONLY normal/inflammation, and
         >= --buffer-um from any neoplastic square -> negative (weight 0.5)
         annotated, any neoplastic class           -> excluded:neoplastic_label
         annotated, only artifact/resection_edge   -> excluded:artifact_neoplastic_slide
         unannotated                               -> excluded:unannotated_neoplastic_slide

Interpretation choices (flagged for Adam in doc.md): ">= 500 um from any
annotation" is read as "from any NEOPLASTIC annotation" (a normal tile sits
inside its own normal annotation, so the literal reading is impossible);
the neoplastic buffer set = {adenocarcinoma, suspicious_for_invasion,
tumor_necrosis, lowgrade_dysplasia, highgrade_dysplasia} for both
neoplastic categories.

Outputs (NEW dir; nothing existing is touched):
    preprocessing/training_tiles/{split}.parquet   (slide_id, x, y, split,
                                                    label, weight,
                                                    negative_type, clarity,
                                                    fold, blur, artifacts)
    preprocessing/training_tiles/excluded.parquet  (slide_id, x, y, split,
                                                    dys, reason)
    + a printed before/after table (positives/negatives per split, exclusions
      by reason, positives lost to the blur cut).

Usage:
    .venv/bin/python scripts/make_training_tiles.py
    .venv/bin/python scripts/make_training_tiles.py --blur-threshold 0.3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
ALLCLASSES = PROJECT / "preprocessing" / "tiling_200_allclasses" / "tiles"
ANNOT = PROJECT / "preprocessing" / "annotations"
SLIDES = PROJECT / "preprocessing" / "tiling_200" / "slides" / "slides.parquet"
METADATA = Path(
    "/mnt/data/Public/Public/colorectum/TCIA_HunCRC/slide_level_annotations.csv"
)
OUT = PROJECT / "preprocessing" / "training_tiles"

NEOPLASTIC = [
    "adenocarcinoma",
    "suspicious_for_invasion",
    "tumor_necrosis",
    "lowgrade_dysplasia",
    "highgrade_dysplasia",
]
NORMALISH = ["normal", "inflammation"]
ROBUST = ["artifact", "resection_edge"]
#: polyp types that disqualify a non_neoplastic slide as a negative source
BAD_POLYP_TYPES = {
    "sessile_serrated",
    "traditional_serrated",
    "tubular_adenoma",
    "tubulovillous_adenoma",
}
SAFE_CATEGORIES = {"negative", "non_neoplastic_lesion"}
NEOPLASTIC_CATEGORIES = {"adenoma", "CRC"}


def _slide_meta() -> pd.DataFrame:
    m = pd.read_csv(METADATA, dtype={"slideID": str})
    m["slideID"] = m["slideID"].str.zfill(3)
    assert len(m) == m.slideID.nunique() == 200
    return m.set_index("slideID")


def _neoplastic_boxes(stem: str):
    """Level-0 boxes of the slide's neoplastic label squares (None if the
    slide has none)."""
    items = json.loads((ANNOT / f"{stem}.json").read_text())["items"]
    pts = np.array(
        [
            (c[0][0], c[0][1], c[1][0] - c[0][0])
            for it in items
            if it["name"] in NEOPLASTIC
            for c in [it["coordinates"]]
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    if not len(pts):
        return None
    return shapely.box(pts[:, 0], pts[:, 1], pts[:, 0] + pts[:, 2], pts[:, 1] + pts[:, 2])


def assign_policy(
    t: pd.DataFrame,
    category: str,
    polyp_type: str,
    *,
    blur_t: float,
    artifact_t: float,
    near_buffer: np.ndarray | None,
    robust_negatives: bool,
    neoplastic_neg_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The whole label policy as one pure function.

    Returns (status, y, weight, negative_type, dys) arrays aligned to t.
    `near_buffer` = per-tile bool, True when the tile is closer than the
    buffer distance to any neoplastic square (None for safe categories).
    """
    n = len(t)
    status = np.empty(n, dtype=object)
    y = np.zeros(n, dtype=np.int8)
    weight = np.ones(n, dtype=np.float32)
    ntype = np.full(n, None, dtype=object)
    A = {c: t[f"annot_{c}"].to_numpy() for c in NEOPLASTIC + NORMALISH + ROBUST}
    dys = (A["lowgrade_dysplasia"] > 0) | (A["highgrade_dysplasia"] > 0)
    neo = np.zeros(n, dtype=bool)
    for c in NEOPLASTIC:
        neo |= A[c] > 0
    norm = np.zeros(n, dtype=bool)
    for c in NORMALISH:
        norm |= A[c] > 0
    robust = np.zeros(n, dtype=bool)
    for c in ROBUST:
        robust |= A[c] > 0
    blur = t["blur"].to_numpy() > blur_t
    art = t["artifacts"].to_numpy() > artifact_t

    pos = dys & ~blur & ~art
    y[pos] = 1
    status[pos] = "positive"

    if category in SAFE_CATEGORIES:
        if category == "non_neoplastic_lesion" and polyp_type in BAD_POLYP_TYPES:
            status[~pos] = "excluded:polyp_type"
            return status, y, weight, ntype, dys
        ok_plain = ~pos & ~neo & ~blur & ~art
        ok_robust = robust & ~neo & ~norm & ~pos & ~blur & ~art
        if not robust_negatives:
            ok_robust = np.zeros(n, dtype=bool)
        neg = ok_plain | ok_robust
        y[neg] = 0
        ntype[neg] = np.where(norm[neg], "normalish", "robust")
        status[neg] = "negative"
        bad = ~pos & ~neg
        status[bad] = np.where(
            blur[bad], "excluded:blur",
            np.where(art[bad], "excluded:qc_artifact", "excluded:neoplastic_label"),
        )
        return status, y, weight, ntype, dys

    #: adenoma / CRC
    near = near_buffer if near_buffer is not None else np.zeros(n, dtype=bool)
    annotated = norm | neo | robust
    ok_neg = annotated & norm & ~neo & ~robust & ~near & ~blur & ~art
    y[ok_neg] = 0
    weight[ok_neg] = neoplastic_neg_weight
    ntype[ok_neg] = "normalish_neoplastic_slide"
    status[ok_neg] = "negative"

    rem = ~pos & ~ok_neg
    status[rem] = np.where(
        blur[rem], "excluded:blur",
        np.where(
            art[rem], "excluded:qc_artifact",
            np.where(
                neo[rem], "excluded:neoplastic_label",
                np.where(
                    robust[rem] & ~norm[rem],
                    "excluded:artifact_neoplastic_slide",
                    np.where(
                        annotated[rem] & norm[rem],
                        "excluded:buffer",
                        "excluded:unannotated_neoplastic_slide",
                    ),
                ),
            ),
        ),
    )
    return status, y, weight, ntype, dys


def build(
    blur_t: float,
    artifact_t: float,
    buffer_um: float,
    robust_negatives: bool,
    neoplastic_neg_weight: float,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = _slide_meta()
    sl = pd.read_parquet(SLIDES)
    sl["stem"] = sl["path"].map(lambda p: Path(p).stem)
    sl = sl.set_index("stem")

    keep, excl = [], []
    for f in sorted(ALLCLASSES.glob("*.parquet")):
        stem = f.stem
        t = pd.read_parquet(f)
        row = sl.loc[stem]
        category = meta.loc[stem, "CATEGORY"]
        polyp = meta.loc[stem, "Polyp type"]
        assert category in SAFE_CATEGORIES | NEOPLASTIC_CATEGORIES, category

        near = None
        if category in NEOPLASTIC_CATEGORIES:
            boxes = _neoplastic_boxes(stem)
            if boxes is not None:
                ds = float(row.downsample)
                buf_px = buffer_um / (float(row.mpp_x) * ds)
                tx = t["x"].to_numpy().astype(np.float64)
                ty = t["y"].to_numpy().astype(np.float64)
                tb = shapely.box(
                    tx * ds, ty * ds,
                    (tx + float(row.tile_extent_x)) * ds,
                    (ty + float(row.tile_extent_y)) * ds,
                )
                #: shapely 2.1 returns a (2, n_pairs) list: row 0 = query
                #: (tile) indices, row 1 = tree (box) indices
                hits = STRtree(boxes).query(
                    tb, predicate="dwithin", distance=buf_px
                )
                near = np.zeros(len(tb), dtype=bool)
                if hits.size:
                    near[hits[0]] = True

        status, y, weight, ntype, dys = assign_policy(
            t, category, polyp,
            blur_t=blur_t, artifact_t=artifact_t, near_buffer=near,
            robust_negatives=robust_negatives,
            neoplastic_neg_weight=neoplastic_neg_weight,
        )
        is_excl = np.fromiter(
            (s.startswith("excluded:") for s in status),
            dtype=bool, count=len(t),
        )
        out = t[["slide_id", "x", "y", "split", "clarity", "fold",
                 "blur", "artifacts"]].copy()
        out["status"] = status
        out["label"] = y
        out["weight"] = weight
        out["negative_type"] = ntype
        keep.append(out[~is_excl])
        e = out[is_excl].copy()
        e["reason"] = [s.split(":", 1)[1] for s in status[is_excl]]
        e["dys"] = dys[is_excl].astype(int)
        excl.append(e[["slide_id", "x", "y", "split", "dys", "reason"]])

    df = pd.concat(keep, ignore_index=True)
    ex = pd.concat(excl, ignore_index=True)
    for split in sorted(df["split"].unique()):
        df[df["split"] == split].to_parquet(OUT / f"{split}.parquet", index=False)
    ex.to_parquet(OUT / "excluded.parquet", index=False)

    print("=== v2 training table ===")
    g = df.groupby("split").agg(
        tiles=("label", "size"), pos=("label", "sum"),
        neg=("negative_type", lambda s: int(s.notna().sum())),
    )
    print(g.to_string())
    print(f"\npositives: {int(df['label'].sum()):,} | "
          f"negatives: {int(df['negative_type'].notna().sum()):,} | "
          f"excluded: {len(ex):,} / {len(df) + len(ex):,} tiles")
    print("\nnegative_type x split:")
    print(
        df[df.negative_type.notna()]
        .groupby(["negative_type", "split"]).size()
        .unstack(fill_value=0).to_string()
    )
    print("\nexcluded by reason:")
    print(ex["reason"].value_counts().to_string())
    print(f"\npositives lost to the blur cut: {int(((ex['reason'] == 'blur') & (ex['dys'] == 1)).sum()):,}")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blur-threshold", type=float, default=0.5)
    ap.add_argument("--artifact-threshold", type=float, default=0.5)
    ap.add_argument("--buffer-um", type=float, default=500.0)
    ap.add_argument("--no-robust-negatives", action="store_true",
                    help="drop artifact/resection-edge tiles even on safe slides")
    ap.add_argument("--neoplastic-neg-weight", type=float, default=0.5)
    a = ap.parse_args()
    build(a.blur_threshold, a.artifact_threshold, a.buffer_um,
          not a.no_robust_negatives, a.neoplastic_neg_weight)
