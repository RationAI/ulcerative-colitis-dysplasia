"""Step 7: within-slide evaluation against annotated "normal" squares (test_preliminary).

No retraining: reuses the Step-4a fold models' all-tile scores written by
``prelim_breakdown.py score`` (``<runs>/<v>/fold<k>/prelim_all_scores.npy``) and
each fold's Youden threshold. test_final is never read.

    python scripts/huncrc/within_slide_eval.py --runs R --all-cache A \
        --annotations <preprocessing>/annotations --embeddings <preprocessing>/embeddings/prov-gigapath \
        --tiling T --slide-csv CSV --out O

Definitions (coverage computed exactly from the annotation JSON squares):
- normal tile: CENTRAL HALF fully inside the union of "normal" squares (same convention
  as the tiling's lowgrade_/highgrade_dysplasia columns; NOT tiling_200_allclasses'
  annot_<class> columns, which are full-tile coverage);
- strict normal: normal tile whose FULL tile has zero LG and HG coverage;
- dysplasia tile: label 1 (central-half LG or HG coverage > 0);
- carcinoma tile: central half fully inside the union of adenocarcinoma /
  suspicious_for_invasion / tumor_necrosis squares ("any" = touched, also reported).
Every number is the mean over the 5 fold models at their own thresholds; 95% CIs
bootstrap over slides (1000 resamples).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.huncrc_annotations import (
    CARCINOMA_CLASSES,
    load_squares,
    rect_coverage,
    tile_rects,
)
from ml.data.huncrc_features import load_features
from ml.data.huncrc_labels import (
    LabelConfig,
    excluded_slide_names,
    load_slide_table,
    load_tiles,
    safe_slide_names,
)
from ml.huncrc_eval import slide_bootstrap, weighted_auroc_ap


FOLDS = range(5)
EPS = 1e-9
MIN_PER_CLASS = 50


def fold_scores(runs: Path, v: str) -> tuple[np.ndarray, np.ndarray]:
    scores = np.stack(
        [np.load(runs / v / f"fold{k}" / "prelim_all_scores.npy") for k in FOLDS]
    )
    thr = np.array(
        [
            json.loads((runs / v / f"fold{k}" / "done.json").read_text())["threshold"]
            for k in FOLDS
        ]
    )
    return scores, thr


def slide_masks(squares: dict, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    half, full = tile_rects(x, y, "central_half"), tile_rects(x, y, "full")
    carc = [squares[c] for c in CARCINOMA_CLASSES if c in squares]
    carc = np.concatenate(carc) if carc else None
    return {
        "half_lg": rect_coverage(squares.get("lowgrade_dysplasia"), half),
        "half_hg": rect_coverage(squares.get("highgrade_dysplasia"), half),
        "half_normal": rect_coverage(squares.get("normal"), half),
        "full_lghg": np.maximum(
            rect_coverage(squares.get("lowgrade_dysplasia"), full),
            rect_coverage(squares.get("highgrade_dysplasia"), full),
        ),
        "half_carc": rect_coverage(carc, half),
    }


def ci_dict(p, lo, hi) -> dict:
    return {"value": p, "ci": [lo, hi]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for a in (
        "runs",
        "all-cache",
        "annotations",
        "embeddings",
        "tiling",
        "slide-csv",
        "out",
    ):
        ap.add_argument(f"--{a}", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["a", "b"])
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    cfg = LabelConfig(min_coverage=0.0)
    st = load_slide_table(
        pd.read_parquet(a.tiling / "slides" / "slides.parquet"),
        pd.read_csv(a.slide_csv, dtype=str),
    )
    stn = st.set_index("name")
    safe = safe_slide_names(st, cfg)
    fs = load_features(a.all_cache, expected_config=cfg, mode="all")
    assert set(fs.splits) == {"test_preliminary"}
    names, slide_of = np.unique(fs.slide_names, return_inverse=True)

    # ---- exact coverage for every test_preliminary tile
    cov = {
        k: np.zeros(len(fs))
        for k in ("half_lg", "half_hg", "half_normal", "full_lghg", "half_carc")
    }
    for j, n in enumerate(names):
        m = slide_of == j
        sq = load_squares(a.annotations / f"{n}.json")
        for k, v in slide_masks(sq, fs.x[m], fs.y[m]).items():
            cov[k][m] = v
    # the recomputation must reproduce the tiling's central-half LG/HG columns
    err = max(
        np.abs(cov["half_lg"] - fs.lg).max(), np.abs(cov["half_hg"] - fs.hg).max()
    )
    assert err < 1e-6, (
        f"central-half LG/HG recomputation differs from the tiling by {err}"
    )

    dys = fs.labels == 1
    normal = cov["half_normal"] >= 1 - EPS
    strict = normal & (cov["full_lghg"] == 0)
    carc_full = cov["half_carc"] >= 1 - EPS
    carc_any = cov["half_carc"] > 0
    assert not (normal & dys).any(), "a tile is both normal and dysplasia"
    has_lghg = (
        np.bincount(slide_of, weights=(cov["half_lg"] > 0) | (cov["half_hg"] > 0)) > 0
    )
    cls = np.where(
        has_lghg, "pos", np.where(np.isin(names, list(safe)), "safe", "other")
    )
    category = stn.loc[names, "category"].to_numpy()
    res: dict = {
        "definitions": __doc__.split("Definitions")[1].split("Every number")[0].strip(),
        "recompute_max_abs_diff": float(err),
        "per_slide": {},
        "variants": {},
    }

    # ---- 5d: counts per positive slide (test_prelim) + slides without normal squares (train+prelim)
    for j, n in enumerate(names):
        m = slide_of == j
        res["per_slide"][n] = {
            "class": cls[j], "category": category[j], "polyp_type": stn.loc[n, "polyp_type"],
            "n_tiles": int(m.sum()), "n_dysplasia": int((m & dys).sum()),
            "n_normal": int((m & normal).sum()), "n_strict_normal": int((m & strict).sum()),
            "n_carcinoma_full": int((m & carc_full).sum()), "n_carcinoma_any": int((m & carc_any).sum()),
            "n_label0": int((m & (fs.labels == 0)).sum()),
        }  # fmt: skip
    tp = load_tiles(a.embeddings, st, ("train", "test_preliminary"))
    no_normal, train_counts = [], {}
    excluded = excluded_slide_names(st, cfg)
    for n, g in tp.groupby("name"):
        sq = load_squares(a.annotations / f"{n}.json")
        lghg = (g["lowgrade_dysplasia"] > 0) | (g["highgrade_dysplasia"] > 0)
        if not lghg.any() or n in excluded:
            continue  # not a positive slide (excluded slides, e.g. 043, never are)
        if "normal" not in sq:
            no_normal.append(n)
        mm = slide_masks(sq, g.x.to_numpy(), g.y.to_numpy())
        nn = mm["half_normal"] >= 1 - EPS
        train_counts[n] = {
            "split": g["split"].iloc[0],
            "n_normal": int(nn.sum()),
            "n_strict_normal": int((nn & (mm["full_lghg"] == 0)).sum()),
        }
    res["5d"] = {
        "positive_slides_without_normal_squares": sorted(no_normal),
        "positive_slides": train_counts,
    }

    pos_slides = np.flatnonzero(cls == "pos")
    for v in a.variants:
        scores, thr = fold_scores(a.runs, v)
        pred = thr[:, None] <= scores
        R: dict = {"thresholds": thr.tolist()}

        # ---- 5a within-slide discrimination
        R["5a"] = {}
        for nname, nmask in (("normal", normal), ("strict_normal", strict)):
            slides_with = [j for j in pos_slides if (nmask & (slide_of == j)).any()]
            items = np.flatnonzero(np.isin(slide_of, slides_with) & (dys | nmask))
            yi = dys[items].astype(int)
            s_idx, inv = np.unique(slide_of[items], return_inverse=True)

            def auc_stat(w, items=items, yi=yi, scores=scores):
                return float(
                    np.mean([weighted_auroc_ap(yi, s[items], w)[0] for s in scores])
                )

            def ap_stat(w, items=items, yi=yi, scores=scores):
                return float(
                    np.mean([weighted_auroc_ap(yi, s[items], w)[1] for s in scores])
                )

            neg_items = items[yi == 0]
            neg_inv = inv[yi == 0]

            def spec_stat(w, neg_items=neg_items, pred=pred):
                return float(
                    np.mean([(~p[neg_items] * w).sum() / w.sum() for p in pred])
                )

            strata = np.zeros(len(s_idx))
            per_slide, left_out = {}, []
            for j in slides_with:
                mj = slide_of == j
                nd, nn = int((mj & dys).sum()), int((mj & nmask).sum())
                if nd < MIN_PER_CLASS or nn < MIN_PER_CLASS:
                    left_out.append(
                        {"slide": names[j], "n_dysplasia": nd, "n_normal": nn}
                    )
                    continue
                it = np.flatnonzero(mj & (dys | nmask))
                per_slide[names[j]] = float(
                    np.mean(
                        [
                            weighted_auroc_ap(dys[it].astype(int), s[it])[0]
                            for s in scores
                        ]
                    )
                )
            R["5a"][nname] = {
                "slides_used": [names[j] for j in slides_with],
                "slides_without_this_normal": [names[j] for j in pos_slides if j not in slides_with],
                "n_dysplasia": int(yi.sum()), "n_normal": int((yi == 0).sum()),
                "auroc": ci_dict(*slide_bootstrap(inv, strata, auc_stat, a.n_boot, 11)),
                "auprc": ci_dict(*slide_bootstrap(inv, strata, ap_stat, a.n_boot, 12)),
                "auprc_baseline": float(yi.mean()),
                "specificity_on_normal": ci_dict(
                    *slide_bootstrap(neg_inv, strata, spec_stat, a.n_boot, 13)
                ),
                "per_slide_auroc": per_slide,
                "per_slide_auroc_median": float(np.median(list(per_slide.values()))) if per_slide else None,
                "per_slide_left_out": left_out,
            }  # fmt: skip

        # ---- 5b slide-type shortcut test
        def group_stats(mask, slide_set, seed, scores=scores, pred=pred):
            items = np.flatnonzero(mask & np.isin(slide_of, slide_set))
            s_idx, inv = np.unique(slide_of[items], return_inverse=True)

            def frac(w, items=items, pred=pred):
                return float(np.mean([(p[items] * w).sum() / w.sum() for p in pred]))

            per = {}
            for j in s_idx:
                it = items[slide_of[items] == j]
                per[names[j]] = {
                    "n": len(it),
                    "median": float(np.mean([np.median(s[it]) for s in scores])),
                    "frac_above": float(np.mean([p[it].mean() for p in pred])),
                }
            meds = [d["median"] for d in per.values()]
            return {
                "n_tiles": len(items), "n_slides": len(s_idx),
                "median_score": float(np.mean([np.median(s[items]) for s in scores])),
                "frac_above": ci_dict(*slide_bootstrap(inv, np.zeros(len(s_idx)), frac, a.n_boot, seed)),
                "per_slide": per,
                "per_slide_median_of_medians": float(np.median(meds)),
                "per_slide_median_range": [float(min(meds)), float(max(meds))],
            }  # fmt: skip

        safe_slides = np.flatnonzero(cls == "safe")
        R["5b"] = {
            "i_normal_in_positive": group_stats(normal, pos_slides, 21),
            "i_strict_normal_in_positive": group_stats(strict, pos_slides, 22),
            "ii_normal_in_safe": group_stats(normal, safe_slides, 23),
            "iii_label0_in_safe": group_stats(fs.labels == 0, safe_slides, 24),
        }

        # ---- 5c carcinoma vs normal in CRC slides with carcinoma squares
        crc = [
            j
            for j in range(len(names))
            if category[j] == "CRC" and (carc_any & (slide_of == j)).any()
        ]
        R["5c"] = {"slides": [names[j] for j in crc], "per_slide": {}}
        for j in crc:
            mj = slide_of == j
            row = {}
            for g, m in (
                ("carcinoma_full", carc_full),
                ("carcinoma_any", carc_any),
                ("normal", normal),
            ):
                it = np.flatnonzero(mj & m)
                row[g] = {
                    "n": len(it),
                    "frac_above": float(np.mean([p[it].mean() for p in pred]))
                    if len(it)
                    else None,
                    "median": float(np.mean([np.median(s[it]) for s in scores]))
                    if len(it)
                    else None,
                }
            R["5c"]["per_slide"][names[j]] = row
        for g, m in (
            ("carcinoma_full", carc_full),
            ("carcinoma_any", carc_any),
            ("normal", normal),
        ):
            it = np.flatnonzero(np.isin(slide_of, crc) & m)
            R["5c"][f"pooled_{g}"] = {
                "n": len(it),
                "frac_above": float(np.mean([p[it].mean() for p in pred]))
                if len(it)
                else None,
            }
        res["variants"][v] = R
        print(f"variant {v} done", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "within_slide.json").write_text(json.dumps(res, indent=2, default=float))
    print(a.out / "within_slide.json")


if __name__ == "__main__":
    main()
