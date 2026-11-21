"""Step 5: harder evaluation of the saved linear-head fold models on test_preliminary.

No retraining. Reuses ``linear_cv.py`` runs (best epoch, scaler, Youden
threshold of each (variant, fold)). test_final is never read.

    # 1) score EVERY test_preliminary tile with each fold model (resumable)
    python scripts/huncrc/prelim_breakdown.py score  --runs R --all-cache A --prelim-cache P
    # 2) breakdowns + slide-level evaluation -> <out>/breakdown.json, breakdown.txt
    python scripts/huncrc/prelim_breakdown.py report --runs R --all-cache A --prelim-cache P \
        --tiling T --slide-csv CSV --out O

5a (labelled tiles): per-slide table, specificity on hyperplastic vs
Not_applicable negative slides, sensitivity on adenoma vs CRC slides and on
LG-only vs HG tiles. 5b (all tiles of all 30 slides): slide AUROC for three
aggregations (mean, mean of top 1%, fraction above threshold), and a
diagnostic of annotated vs unannotated tiles inside positive slides (NOT a
metric: unannotated tiles are not ground-truth negatives), with the distance
of high-scoring unannotated tiles to the nearest annotated tile.

Every number is the mean over the 5 fold models, each at its own validation
threshold; 95% CIs bootstrap over slides (1000 resamples, stratified by slide
class where a statistic mixes classes).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from ml.data.huncrc_features import load_features
from ml.data.huncrc_labels import LabelConfig, load_slide_table, safe_slide_names
from ml.huncrc_eval import slide_bootstrap, weighted_auroc_ap
from ml.huncrc_train import Scaler, make_head, predict


VARIANTS = ("a", "b")
FOLDS = range(5)
STRIDE = 112  # level-2 px between tile origins
TOP_FRACTION = 0.01


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_dir(runs: Path, v: str, k: int) -> Path:
    return runs / v / f"fold{k}"


def labelled_index(all_fs, lab_fs) -> np.ndarray:
    """Row of the all-tile cache for every row of the labelled cache."""
    a = pd.DataFrame({"n": all_fs.slide_names, "x": all_fs.x, "y": all_fs.y})
    a["row"] = np.arange(len(a))
    b = pd.DataFrame({"n": lab_fs.slide_names, "x": lab_fs.x, "y": lab_fs.y})
    m = b.merge(a, on=["n", "x", "y"], how="left", validate="1:1")
    assert m["row"].notna().all(), "labelled tile missing from the all-tile cache"
    return m["row"].to_numpy(np.int64)


def cmd_score(a) -> None:
    cfg = LabelConfig(min_coverage=0.0)
    all_fs = load_features(a.all_cache, expected_config=cfg, mode="all")
    lab = load_features(a.prelim_cache, expected_config=cfg)
    assert set(all_fs.splits) == {"test_preliminary"} == set(lab.splits)
    idx = labelled_index(all_fs, lab)
    assert np.array_equal(all_fs.labels[idx], lab.labels)
    rows = np.arange(len(all_fs))
    for v in a.variants:
        for k in FOLDS:
            d = run_dir(a.runs, v, k)
            out = d / "prelim_all_scores.npy"
            if out.is_file():
                log(f"{v}/fold{k}: scored, skipped")
                continue
            done = json.loads((d / "done.json").read_text())
            z = np.load(d / "scaler.npz")
            sc = Scaler(z["mean"], z["std"])
            model = make_head("linear", all_fs.features.shape[1])
            state = torch.load(
                d / f"epoch_{done['best_epoch'] - 1:02d}.pt", weights_only=False
            )
            model.load_state_dict(state["model"])
            s = predict(model, all_fs.features, rows, sc)
            # the labelled tiles must reproduce the step-4a scores exactly
            ref = np.load(d / "prelim_scores.npy")
            err = float(np.abs(s[idx] - ref).max())
            assert err < 1e-5, f"{v}/fold{k}: labelled scores differ by {err}"
            tmp = out.with_name(out.name + ".tmp")
            with open(tmp, "wb") as f:
                np.save(f, s)
            os.replace(tmp, out)
            log(f"{v}/fold{k}: {len(s)} tiles scored (labelled max |diff| {err:.1e})")


# --------------------------------------------------------------------------- report


def _fold_scores(runs: Path, v: str) -> tuple[np.ndarray, np.ndarray]:
    S = np.stack(
        [np.load(run_dir(runs, v, k) / "prelim_all_scores.npy") for k in FOLDS]
    )
    thr = np.array(
        [
            json.loads((run_dir(runs, v, k) / "done.json").read_text())["threshold"]
            for k in FOLDS
        ]
    )
    return S, thr


def _tile_rate(items, slide_of, pred, n_boot, seed):
    """Fold-averaged fraction of ``pred`` over ``items``, CI over their slides."""
    if len(items) == 0:
        return {"value": None, "ci": [None, None], "n_tiles": 0, "n_slides": 0}
    slides, inv = np.unique(slide_of[items], return_inverse=True)
    P = pred[:, items].astype(float)

    def stat(w):
        return float(np.mean(P @ w / w.sum()))

    p, lo, hi = slide_bootstrap(inv, np.zeros(len(slides)), stat, n_boot, seed)
    return {
        "value": p,
        "ci": [lo, hi],
        "n_tiles": len(items),
        "n_slides": len(slides),
    }


def _slide_auc(y, scores, strata, n_boot, seed):
    """Fold-averaged slide AUROC of slide ``scores`` (folds x slides)."""

    def stat(w):
        return float(np.mean([weighted_auroc_ap(y, x, w)[0] for x in scores]))

    p, lo, hi = slide_bootstrap(np.arange(len(y)), strata, stat, n_boot, seed)
    return {
        "value": p,
        "ci": [lo, hi],
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
    }


def cmd_report(a) -> None:
    cfg = LabelConfig(min_coverage=0.0)
    fs = load_features(a.all_cache, expected_config=cfg, mode="all")
    st = load_slide_table(
        pd.read_parquet(a.tiling / "slides" / "slides.parquet"),
        pd.read_csv(a.slide_csv, dtype=str),
    )
    st = st.set_index("name")
    names, slide_of = np.unique(fs.slide_names, return_inverse=True)
    assert set(st.loc[names, "split"]) == {"test_preliminary"}
    safe = safe_slide_names(st.reset_index(), cfg)
    annotated = fs.coverage > 0
    has_lghg = np.bincount(slide_of, weights=annotated, minlength=len(names)) > 0
    cls = np.where(
        has_lghg, "pos", np.where(np.isin(names, list(safe)), "safe", "other")
    )
    lab = fs.labels >= 0
    res: dict = {
        "slides": {},
        "variants": {},
        "n_slides": {c: int((cls == c).sum()) for c in ("pos", "safe", "other")},
    }

    for v in a.variants:
        S, thr = _fold_scores(a.runs, v)
        pred = thr[:, None] <= S
        R: dict = {"thresholds": thr.tolist()}

        # ---- 5a: labelled tiles
        neg = lab & (fs.labels == 0)
        pos = lab & (fs.labels == 1)
        ptype = st.loc[names, "polyp_type"].to_numpy()[slide_of]
        cat = st.loc[names, "category"].to_numpy()[slide_of]
        lg_only = pos & (fs.hg == 0)
        hg = pos & (fs.hg > 0)
        groups = {
            "spec_hyperplastic": (neg & (ptype == "hyperplastic"), False),
            "spec_not_applicable": (neg & (ptype == "Not_applicable"), False),
            "spec_all_negatives": (neg, False),
            "sens_adenoma_slides": (pos & (cat == "adenoma"), True),
            "sens_crc_slides": (pos & (cat == "CRC"), True),
            "sens_lg_only_tiles": (lg_only, True),
            "sens_hg_tiles": (hg, True),
            "sens_all_positives": (pos, True),
        }
        R["5a"] = {}
        for gname, (mask, is_sens) in groups.items():
            items = np.flatnonzero(mask)
            rate = pred if is_sens else ~pred
            R["5a"][gname] = _tile_rate(items, slide_of, rate, a.n_boot, seed=1)

        # ---- 5b: all tiles, slide aggregations
        n_per = np.bincount(slide_of)
        mean_all = np.stack([np.bincount(slide_of, weights=s) / n_per for s in S])
        frac_all = np.stack([np.bincount(slide_of, weights=p) / n_per for p in pred])
        top = np.empty_like(mean_all)
        for j in range(len(names)):
            m = slide_of == j
            kk = max(1, math.ceil(TOP_FRACTION * m.sum()))
            for f in range(len(S)):
                top[f, j] = np.sort(S[f, m])[-kk:].mean()
        aggs = {"mean_all": mean_all, "top1pct_mean": top, "frac_above_thr": frac_all}
        R["5b_slide_auroc"] = {}
        main = cls != "other"
        y_main = (cls[main] == "pos").astype(int)
        y_sec = (cls != "safe").astype(int)
        strata_main = y_main
        strata_sec = np.select([cls == "pos", cls == "safe"], [0, 1], 2)
        for an, X in aggs.items():
            R["5b_slide_auroc"][an] = {
                "pos_vs_safe": _slide_auc(y_main, X[:, main], strata_main, a.n_boot, 2),
                "pos_plus_crc4_vs_safe": _slide_auc(y_sec, X, strata_sec, a.n_boot, 3),
            }

        # ---- 5b: diagnostic inside positive slides (NOT a metric)
        in_pos = cls[slide_of] == "pos"
        ann, unann = in_pos & annotated, in_pos & ~annotated
        R["5b_diag"] = {
            g: {
                "n_tiles": int(m.sum()),
                "median_score": float(np.mean([np.median(s[m]) for s in S])),
                "frac_above_thr": float(np.mean([p[m].mean() for p in pred])),
            }
            for g, m in (("annotated_lghg", ann), ("unannotated", unann))
        }

        # ---- 5b optional: where are the high-scoring unannotated tiles?
        bins = [
            (0.0, 1.5, "adjacent (<=1 step)"),
            (1.5, 5.5, "2-5 steps"),
            (5.5, np.inf, ">5 steps"),
        ]
        dist = np.full(len(fs), np.nan)
        for j in np.flatnonzero(cls == "pos"):
            m = slide_of == j
            a_xy = np.stack([fs.x[m & annotated], fs.y[m & annotated]], 1) / STRIDE
            u = np.flatnonzero(m & ~annotated)
            if len(u):
                dd, _ = cKDTree(a_xy).query(np.stack([fs.x[u], fs.y[u]], 1) / STRIDE)
                dist[u] = dd
        loc = {"all_unannotated": {}, "unannotated_above_thr": {}}
        for lo_, hi_, label in bins:
            inb = unann & (dist > lo_) & (dist <= hi_)
            loc["all_unannotated"][label] = float(inb.sum() / unann.sum())
            loc["unannotated_above_thr"][label] = float(
                np.mean([(inb & p).sum() / max((unann & p).sum(), 1) for p in pred])
            )
        loc["n_unannotated_above_thr_mean"] = float(
            np.mean([(unann & p).sum() for p in pred])
        )
        R["5b_location"] = loc

        # ---- per-slide table rows
        for j, n in enumerate(names):
            m = slide_of == j
            ml = m & lab
            row = res["slides"].setdefault(
                n,
                {
                    "category": st.loc[n, "category"],
                    "polyp_type": st.loc[n, "polyp_type"],
                    "class": cls[j],
                    "n_tiles": int(m.sum()),
                    "n_labelled": int(ml.sum()),
                    "n_annotated": int((m & annotated).sum()),
                },
            )
            row[f"{v}_lab_mean"] = (
                float(np.mean([s[ml].mean() for s in S])) if ml.any() else None
            )
            row[f"{v}_lab_frac"] = (
                float(np.mean([p[ml].mean() for p in pred])) if ml.any() else None
            )
            row[f"{v}_all_mean"] = float(mean_all[:, j].mean())
            row[f"{v}_top1"] = float(top[:, j].mean())
            row[f"{v}_all_frac"] = float(frac_all[:, j].mean())
            mu = m & ~annotated
            row[f"{v}_unann_frac"] = (
                float(np.mean([p[mu].mean() for p in pred]))
                if cls[j] == "pos" and mu.any()
                else None
            )
        res["variants"][v] = R
        log(f"report {v} done")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "breakdown.json").write_text(json.dumps(res, indent=2, default=float))
    text = render(res)
    (a.out / "breakdown.txt").write_text(text)
    print(text)


def _f(x, nd=3):
    return "  -  " if x is None else f"{x:.{nd}f}"


def render(res: dict) -> str:
    L = []
    order = sorted(
        res["slides"].items(),
        key=lambda kv: ({"pos": 0, "other": 1, "safe": 2}[kv[1]["class"]], kv[0]),
    )
    L.append(
        "Per-slide (test_preliminary). lab = labelled tiles, all = all tiles; mean over 5 fold models"
    )
    L.append(
        "slide class  category               polyp            n_all  n_lab  n_ann | "
        "a:lab_mean lab>thr all_mean top1% all>thr unann>thr | b:lab_mean lab>thr all_mean top1% all>thr unann>thr"
    )
    for n, r in order:
        L.append(
            f"{n}   {r['class']:<5}  {r['category']:<21}  {r['polyp_type']:<15}  {r['n_tiles']:>6} {r['n_labelled']:>6} {r['n_annotated']:>6} | "
            + "  ".join(
                _f(r.get(f"a_{c}"))
                for c in (
                    "lab_mean",
                    "lab_frac",
                    "all_mean",
                    "top1",
                    "all_frac",
                    "unann_frac",
                )
            )
            + " | "
            + "  ".join(
                _f(r.get(f"b_{c}"))
                for c in (
                    "lab_mean",
                    "lab_frac",
                    "all_mean",
                    "top1",
                    "all_frac",
                    "unann_frac",
                )
            )
        )
    for v, R in res["variants"].items():
        L.append(
            f"\nVariant ({v}). thresholds per fold: "
            + ", ".join(f"{t:.3f}" for t in R["thresholds"])
        )
        L.append("5a labelled tiles            value  [95% CI]        tiles  slides")
        for g, d in R["5a"].items():
            if d["value"] is None:
                L.append(f"  {g:<26} n/a (no tiles)")
                continue
            L.append(
                f"  {g:<26} {d['value']:.3f}  [{d['ci'][0]:.3f}-{d['ci'][1]:.3f}]  {d['n_tiles']:>7}  {d['n_slides']:>3}"
            )
        n = res["n_slides"]
        L.append(
            f"5b slide AUROC (all tiles)   pos vs safe ({n['pos']}/{n['safe']})        "
            f"pos+other vs safe ({n['pos'] + n['other']}/{n['safe']})"
        )
        for an, d in R["5b_slide_auroc"].items():
            m, s = d["pos_vs_safe"], d["pos_plus_crc4_vs_safe"]
            L.append(
                f"  {an:<26} {m['value']:.3f} [{m['ci'][0]:.3f}-{m['ci'][1]:.3f}]      {s['value']:.3f} [{s['ci'][0]:.3f}-{s['ci'][1]:.3f}]"
            )
        L.append(
            "5b diagnostic in positive slides (NOT a metric)   tiles   median score  frac>thr"
        )
        for g, d in R["5b_diag"].items():
            L.append(
                f"  {g:<46} {d['n_tiles']:>7}   {d['median_score']:.3f}        {d['frac_above_thr']:.3f}"
            )
        loc = R["5b_location"]
        L.append(
            f"5b distance of unannotated tiles to nearest annotated tile (positive slides); "
            f"mean # above thr: {loc['n_unannotated_above_thr_mean']:.0f}"
        )
        for b in loc["all_unannotated"]:
            L.append(
                f"  {b:<22} all unannotated {loc['all_unannotated'][b]:.3f}   above thr {loc['unannotated_above_thr'][b]:.3f}"
            )
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["score", "report"])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--all-cache", type=Path, required=True)
    p.add_argument("--prelim-cache", type=Path, required=True)
    p.add_argument("--tiling", type=Path)
    p.add_argument("--slide-csv", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.cmd == "score":
        cmd_score(a)
    else:
        if not (a.tiling and a.slide_csv and a.out):
            p.error("report needs --tiling, --slide-csv and --out")
        cmd_report(a)


if __name__ == "__main__":
    main()
