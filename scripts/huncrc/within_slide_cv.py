"""Step 8: linear head with within-slide negatives, 5-fold CV + Step-7 evaluation.

Labels (cache built with ``build_feature_cache.py --within-slide-negatives
--exclude-carcinoma``): positives = dysplasia tiles; negatives = safe-slide
negatives + strict-normal tiles of positive slides; carcinoma tiles and the
excluded slides never train. Negatives are capped per slide in TRAINING only
(validation keeps every tile).

    # resumable: finished (variant, fold) runs are skipped, an interrupted run
    # resumes after its last complete epoch
    python scripts/huncrc/within_slide_cv.py train --train-cache C --prelim-all-cache P --runs R
    # report (works on partial results), next to the saved Step-4a models (A)
    python scripts/huncrc/within_slide_cv.py report --train-cache C --prelim-all-cache P \
        --runs R --a-runs <step4a linear_cv dir> --annotations A --tiling T --slide-csv CSV

Variants: B = new labels, unweighted BCE; C = new labels, class-balanced BCE.
A-unw / A-bal = the Step-4a variants (a) / (b), reused without retraining.
Same protocol as Step 4a: scaler from the training folds only, up to N epochs,
best epoch = max validation AUPRC, Youden threshold on the validation fold,
seed = fold index. test_final is never read; no MLflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ml.data.huncrc_features import FeatureSet, load_features
from ml.data.huncrc_labels import LabelConfig, load_slide_table, safe_slide_names
from ml.huncrc_eval import evaluate, weighted_auroc_ap, youden_threshold
from ml.huncrc_train import (
    Scaler,
    TrainConfig,
    balanced_weights,
    fit_epochs,
    make_head,
    predict,
)
from ml.huncrc_wsmetrics import prelim_masks, prelim_report


VARIANTS = {"B": None, "C": "balanced"}
A_VARIANTS = {"A-unw": "a", "A-bal": "b"}
NEW_RULES = LabelConfig(within_slide_negatives=True, exclude_carcinoma=True)
OLD_RULES = LabelConfig()
N_NULL = 10


def available_cpus() -> int:
    """CPUs this process may really use: cgroup quota, else CPU affinity.

    ``os.cpu_count()`` reports the host (e.g. 192) even when the pod is limited
    to 1 CPU, which would oversubscribe torch and BLAS threads.
    """
    n = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save(obj: Any, path: Path, kind: str = "json") -> None:
    """Atomic write (tmp + rename), so an interrupted run never leaves half files."""
    tmp = path.with_name(path.name + ".tmp")
    if kind == "npy":
        with open(tmp, "wb") as f:
            np.save(f, obj)
    elif kind == "pt":
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def subsample_per_slide(
    rows: np.ndarray, names: np.ndarray, cap: int | None, rng: np.random.Generator
) -> np.ndarray:
    """At most ``cap`` rows per slide (random, sorted output); None = keep all."""
    if cap is None:
        return np.sort(rows)
    keep = []
    for n in np.unique(names[rows]):
        r = rows[names[rows] == n]
        keep.append(r if len(r) <= cap else rng.choice(r, cap, replace=False))
    return np.sort(np.concatenate(keep)) if keep else rows[:0]


def neg_source(fs: FeatureSet, safe: set[str]) -> np.ndarray:
    """'safe_slide' / 'positive_slide_normal' for label-0 rows, '' otherwise."""
    is_safe = np.isin(fs.slide_names, list(safe))
    out = np.full(len(fs), "", dtype=object)
    out[(fs.labels == 0) & is_safe] = "safe_slide"
    out[(fs.labels == 0) & ~is_safe] = "positive_slide_normal"
    return out


def fold_rows(
    fs: FeatureSet,
    k: int,
    smoke: int | None,
    neg_cap: int | None,
    cap_seed: int,
    safe: set[str],
):
    tr = np.flatnonzero((fs.folds >= 0) & (fs.folds != k))
    val = np.flatnonzero(fs.folds == k)
    assert not set(fs.slide_names[tr]) & set(fs.slide_names[val]), "slide overlap"
    rng = np.random.default_rng([cap_seed, k])
    if smoke:
        tr = subsample_per_slide(tr, fs.slide_names, smoke, rng)
        val = subsample_per_slide(val, fs.slide_names, smoke, rng)
    pos = tr[fs.labels[tr] == 1]
    neg = subsample_per_slide(tr[fs.labels[tr] == 0], fs.slide_names, neg_cap, rng)
    train = np.sort(np.concatenate([pos, neg]))
    src = neg_source(fs, safe)

    def counts(r: np.ndarray) -> dict[str, int]:
        return {
            "pos": int((fs.labels[r] == 1).sum()),
            "neg_safe_slide": int((src[r] == "safe_slide").sum()),
            "neg_positive_slide_normal": int((src[r] == "positive_slide_normal").sum()),
            "slides": len(set(fs.slide_names[r])),
            "slides_with_pos_slide_normal": len(
                set(fs.slide_names[r][src[r] == "positive_slide_normal"])
            ),
        }

    return (
        train,
        val,
        {"train_before_cap": counts(tr), "train": counts(train), "val": counts(val)},
    )


def train_one(fs, pre, variant, k, d: Path, a) -> None:
    if (d / "done.json").is_file():
        log(f"{variant}/fold{k}: done, skipped")
        return
    d.mkdir(parents=True, exist_ok=True)
    safe = safe_slide_names(a.slide_table, NEW_RULES)
    tr, val, cnt = fold_rows(
        fs, k, a.smoke_tiles_per_slide, a.neg_cap, a.cap_seed, safe
    )
    if (d / "train_rows.npy").is_file():  # resume: rows must be what we trained on
        assert np.array_equal(np.load(d / "train_rows.npy"), tr), (
            "train rows changed; delete the run dir"
        )
        assert np.array_equal(np.load(d / "val_rows.npy"), val), (
            "val rows changed; delete the run dir"
        )
    else:
        save(tr, d / "train_rows.npy", "npy")
        save(val, d / "val_rows.npy", "npy")
        save(
            {
                **cnt,
                "neg_cap": a.neg_cap,
                "cap_seed": a.cap_seed,
                "smoke_tiles_per_slide": a.smoke_tiles_per_slide,
            },
            d / "counts.json",
        )
    x, y = fs.features, fs.labels
    if (d / "scaler.npz").is_file():
        z = np.load(d / "scaler.npz")
        sc = Scaler(z["mean"], z["std"])
    else:
        sc = Scaler.fit(x, tr)
        np.savez(d / "scaler.npz", mean=sc.mean, std=sc.std)
    cw = balanced_weights(y[tr]) if VARIANTS[variant] == "balanced" else None
    cfg = TrainConfig(epochs=a.epochs, seed=k)
    done = sorted(int(p.stem.split("_")[1]) for p in d.glob("epoch_??.pt"))
    resume = (
        torch.load(d / f"epoch_{done[-1]:02d}.pt", weights_only=False) if done else None
    )
    log(
        f"{variant}/fold{k}: start ({'resume after epoch ' + str(resume['epoch'] + 1) if resume else 'new'}); "
        f"train {cnt['train']}, val {cnt['val']}, class weights {cw}"
    )
    t0 = time.time()
    for ep, model, state, loss in fit_epochs("linear", x, y, tr, sc, cfg, cw, resume):
        s = predict(model, x, val, sc)
        auroc, auprc = weighted_auroc_ap(y[val], s)
        save(s, d / f"epoch_{ep:02d}_val.npy", "npy")
        save(
            {
                "epoch": ep + 1,
                "train_loss": loss,
                "val_auroc": auroc,
                "val_auprc": auprc,
                "seconds": time.time() - t0,
            },
            d / f"epoch_{ep:02d}.json",
        )
        save(state, d / f"epoch_{ep:02d}.pt", "pt")  # written last = epoch complete
        log(
            f"{variant}/fold{k} epoch {ep + 1}/{a.epochs}: loss {loss:.4f} val AUROC {auroc:.4f} AUPRC {auprc:.4f} ({time.time() - t0:.0f}s)"
        )
    hist = [
        json.loads((d / f"epoch_{e:02d}.json").read_text()) for e in range(a.epochs)
    ]
    best = int(np.argmax([h["val_auprc"] for h in hist]))
    thr = youden_threshold(y[val], np.load(d / f"epoch_{best:02d}_val.npy"))
    model = make_head("linear", x.shape[1], cfg.dropout)
    model.load_state_dict(
        torch.load(d / f"epoch_{best:02d}.pt", weights_only=False)["model"]
    )
    save(
        predict(model, pre.features, np.arange(len(pre)), sc),
        d / "prelim_all_scores.npy",
        "npy",
    )
    save(
        {
            "variant": variant,
            "fold": k,
            "best_epoch": best + 1,
            "threshold": thr,
            "class_weights": cw,
            "epochs": a.epochs,
            "config": cfg.__dict__,
        },
        d / "done.json",
    )
    log(
        f"{variant}/fold{k}: DONE best epoch {best + 1}, threshold {thr:.4f}, {time.time() - t0:.0f}s"
    )


# ------------------------------------------------------------------------- report


def _model(d: Path, dim: int) -> tuple[torch.nn.Module, Scaler, dict]:
    done = json.loads((d / "done.json").read_text())
    z = np.load(d / "scaler.npz")
    m = make_head("linear", dim)
    m.load_state_dict(
        torch.load(d / f"epoch_{done['best_epoch'] - 1:02d}.pt", weights_only=False)[
            "model"
        ]
    )
    return m, Scaler(z["mean"], z["std"]), done


def report(a) -> dict:
    fs = load_features(a.train_cache, expected_config=NEW_RULES)
    pre = load_features(a.prelim_all_cache, expected_config=OLD_RULES, mode="all")
    safe = safe_slide_names(a.slide_table, NEW_RULES)
    src = neg_source(fs, safe)
    old_label = (fs.labels == 1) | (src == "safe_slide")  # Step-4a label set
    masks = prelim_masks(pre, a.slide_table, a.annotations)
    sets = {name: a.a_runs / v for name, v in A_VARIANTS.items()} | {
        v: a.runs / v for v in VARIANTS
    }
    out: dict[str, Any] = {"model_sets": {}, "counts": {}}
    for name, root in sets.items():
        folds = [k for k in range(5) if (root / f"fold{k}" / "done.json").is_file()]
        if not folds:
            log(f"report: {name}: no finished folds, skipped")
            continue
        R: dict[str, Any] = {"folds": folds, "val": {}}
        prelim_scores, thrs = [], []
        for k in folds:
            d = root / f"fold{k}"
            model, sc, done = _model(d, fs.features.shape[1])
            val = (
                np.load(d / "val_rows.npy")
                if name in VARIANTS
                else np.flatnonzero(fs.folds == k)
            )
            if name in VARIANTS:
                s_val = np.load(d / f"epoch_{done['best_epoch'] - 1:02d}_val.npy")
                out["counts"].setdefault(
                    f"fold{k}", json.loads((d / "counts.json").read_text())
                )
            else:  # Step-4a model scored on the new-label validation rows
                s_val = predict(model, fs.features, val, sc)
            thr = float(done["threshold"])
            ev_new = evaluate(
                fs.slide_names[val], fs.labels[val], s_val, thr, a.n_boot, seed=k
            )
            ol = old_label[val]
            ev_old = evaluate(
                fs.slide_names[val][ol],
                fs.labels[val][ol],
                s_val[ol],
                thr,
                a.n_boot,
                seed=k,
            )
            null = []
            for seed in range(N_NULL):
                torch.manual_seed(seed)
                null.append(
                    weighted_auroc_ap(
                        fs.labels[val],
                        predict(
                            make_head("linear", fs.features.shape[1]),
                            fs.features,
                            val,
                            sc,
                        ),
                    )
                )
            null = np.array(null)
            R["val"][f"fold{k}"] = {
                "best_epoch": done["best_epoch"], "threshold": thr,
                "new": ev_new.point, "new_ci": ev_new.ci, "old": ev_old.point, "old_ci": ev_old.ci,
                "null_auroc": [float(null[:, 0].mean()), float(null[:, 0].std())],
                "null_auprc": [float(null[:, 1].mean()), float(null[:, 1].std())],
            }  # fmt: skip
            prelim_scores.append(np.load(d / "prelim_all_scores.npy"))
            thrs.append(thr)
            log(f"report: {name} fold{k} validation done")
        R["prelim"] = prelim_report(
            np.stack(prelim_scores), np.array(thrs), masks, a.n_boot, seed=0
        )
        out["model_sets"][name] = R
        log(f"report: {name} test_preliminary done ({len(folds)} folds)")
    return out


def render(rep: dict) -> str:
    L = []
    c = rep["counts"]
    if c:
        L.append(
            "Training tiles per validation fold k (after the per-slide negative cap; before cap in brackets)"
        )
        L.append(
            "fold  positives  neg safe-slide        neg positive-slide normal   val pos / safe / pos-slide normal"
        )
        for f, v in sorted(c.items()):
            t, b, va = v["train"], v["train_before_cap"], v["val"]
            L.append(
                f"{f[4:]:>4}  {t['pos']:>9}  {t['neg_safe_slide']:>7} ({b['neg_safe_slide']:>7})   "
                f"{t['neg_positive_slide_normal']:>7} ({b['neg_positive_slide_normal']:>7}) "
                f"{t['slides_with_pos_slide_normal']:>3} sl  {va['pos']:>7} / {va['neg_safe_slide']:>6} / {va['neg_positive_slide_normal']:>6}"
            )
    names = list(rep["model_sets"])

    def ms(name, key, fmt=".3f"):
        s = rep["model_sets"][name]["prelim"]["summary"].get(key)
        if s is None:
            return "-"
        txt = (
            f"{s['mean']:{fmt}}+/-{s['sd']:{fmt}}"
            if np.isfinite(s["sd"])
            else f"{s['mean']:{fmt}}"
        )
        return txt + (
            f" [{s['ci'][0]:{fmt}}-{s['ci'][1]:{fmt}}]"
            if "ci" in s and np.isfinite(s["ci"][0])
            else ""
        )

    L.append(
        "\nVALIDATION (fold k; new labels, and the Step-4a label set 'old'); mean over finished folds"
    )
    L.append(
        "model   folds  new AUROC  new AUPRC  sens   spec   F1    | old AUROC  old AUPRC  sens   spec   F1    | null AUROC"
    )
    for n in names:
        val_res = rep["model_sets"][n]["val"]

        def mean(key, part, val_res=val_res):
            return np.mean([v[part][key] for v in val_res.values()])

        null = np.mean([v["null_auroc"][0] for v in val_res.values()])
        L.append(
            f"{n:<7} {len(val_res):>5}  {mean('auroc', 'new'):.3f}      {mean('auprc', 'new'):.3f}      "
            f"{mean('sensitivity', 'new'):.3f}  {mean('specificity', 'new'):.3f}  {mean('f1', 'new'):.3f} | "
            f"{mean('auroc', 'old'):.3f}      {mean('auprc', 'old'):.3f}      {mean('sensitivity', 'old'):.3f}  "
            f"{mean('specificity', 'old'):.3f}  {mean('f1', 'old'):.3f} | {null:.3f}"
        )
    L.append(
        "\nTEST_PRELIMINARY (each fold model at its own validation threshold): mean +/- sd over folds [95% CI of the mean, slide bootstrap]"
    )
    rows = [
        ("(i)   within-slide AUROC, dysplasia vs strict normal", "ws_auroc"),
        ("      within-slide AUPRC", "ws_auprc"),
        ("      specificity on normal, positive slides", "spec_normal_pos"),
        ("      specificity on strict normal, positive slides", "spec_strict_pos"),
        ("      median per-slide within-slide AUROC", "per_slide_ws_auroc_median"),
        ("(ii)  normal in positive slides: median score", "median_normal_pos"),
        ("      normal in positive slides: frac > thr", "frac_above_normal_pos"),
        ("      normal in safe slides: median score", "median_normal_safe"),
        ("      normal in safe slides: frac > thr", "frac_above_normal_safe"),
        ("      label-0 in safe slides: frac > thr", "frac_above_label0_safe"),
        ("(iii) old-style tile AUROC (dysplasia vs safe neg)", "old_auroc"),
        ("(iv)  slide AUROC, mean of all tiles", "slide_auroc_mean_all"),
        ("      slide AUROC, mean of top 1%", "slide_auroc_top1pct_mean"),
        ("      slide AUROC, fraction > threshold", "slide_auroc_frac_above_thr"),
        ("(v)   carcinoma tiles: frac > thr", "frac_above_carcinoma"),
        ("      normal tiles, same CRC slides: frac > thr", "frac_above_normal_crc"),
    ]
    for label, key in rows:
        L.append(f"{label:<52}" + " | ".join(f"{n}: {ms(n, key)}" for n in names))
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["train", "report"])
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--prelim-all-cache", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--tiling", type=Path, required=True)
    p.add_argument("--slide-csv", type=Path, required=True)
    p.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS)
    )
    p.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--neg-cap", type=int, default=5000, help="max negatives per slide in training"
    )
    p.add_argument("--cap-seed", type=int, default=0)
    p.add_argument(
        "--smoke-tiles-per-slide",
        type=int,
        default=None,
        help="smoke test: max rows per slide",
    )
    p.add_argument(
        "--threads", type=int, default=None, help="default: the cgroup CPU quota"
    )
    p.add_argument(
        "--a-runs", type=Path, help="report: Step-4a linear_cv runs dir (variants a, b)"
    )
    p.add_argument(
        "--annotations", type=Path, help="report: dir of <slide>.json squares"
    )
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, help="report: output dir (default: --runs)")
    a = p.parse_args()
    a.threads = a.threads or available_cpus()
    torch.set_num_threads(a.threads)
    a.slide_table = load_slide_table(
        pd.read_parquet(a.tiling / "slides" / "slides.parquet"),
        pd.read_csv(a.slide_csv, dtype=str),
    )
    log(
        f"{a.cmd}: threads={torch.get_num_threads()} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} python={sys.version.split()[0]}"
    )
    if a.cmd == "train":
        fs = load_features(a.train_cache, expected_config=NEW_RULES)
        pre = load_features(a.prelim_all_cache, expected_config=OLD_RULES, mode="all")
        assert set(fs.splits) == {"train"} and set(pre.splits) == {"test_preliminary"}
        a.runs.mkdir(parents=True, exist_ok=True)
        (a.runs / "run_config.json").write_text(
            json.dumps(
                {k: str(v) for k, v in vars(a).items() if k != "slide_table"}, indent=2
            )
        )
        for v in a.variants:
            for k in a.folds:
                train_one(fs, pre, v, k, a.runs / v / f"fold{k}", a)
        log("train: all requested runs finished")
    else:
        if not (a.a_runs and a.annotations):
            p.error("report needs --a-runs and --annotations")
        rep = report(a)
        out = a.out or a.runs
        out.mkdir(parents=True, exist_ok=True)
        save(rep, out / "report.json")
        text = render(rep)
        (out / "table.txt").write_text(text + "\n")
        print(text, flush=True)
        log(f"report written to {out / 'report.json'} and {out / 'table.txt'}")


if __name__ == "__main__":
    main()
