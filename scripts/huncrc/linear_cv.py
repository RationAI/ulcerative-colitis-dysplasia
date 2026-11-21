"""Step 4a: linear head, 5-fold CV on train, evaluation on test_preliminary.

    # resumable; re-running skips finished (variant, fold) runs and resumes
    # an interrupted one from its last saved epoch
    python scripts/huncrc/linear_cv.py train  --train-cache C --prelim-cache P --runs R
    python scripts/huncrc/linear_cv.py report --train-cache C --prelim-cache P --runs R

Variants: ``a`` unweighted BCE (repo default), ``b`` class-balanced BCE.
Per (variant, fold k): train on the other four folds for up to ``--epochs``,
keep the epoch with the best validation AUPRC, choose the Youden-J threshold
on validation fold k, score test_preliminary with that epoch.

Layout ``R/<variant>/fold<k>/``: ``scaler.npz``, ``val_rows.npy``,
``epoch_XX.pt`` (model + optimizer), ``epoch_XX_val.npy`` (val scores),
``epoch_XX.json`` (per-epoch metrics), ``prelim_scores.npy``, ``done.json``.
No MLflow; test_final is never read (both caches refuse it).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ml.data.huncrc_features import load_features
from ml.data.huncrc_labels import LabelConfig
from ml.huncrc_eval import confusion, evaluate, weighted_auroc_ap, youden_threshold
from ml.huncrc_train import (
    Scaler,
    TrainConfig,
    balanced_weights,
    fit_epochs,
    make_head,
    predict,
)


VARIANTS = {"a": "unweighted", "b": "class-balanced"}
N_NULL = 10


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_save(obj, path: Path, kind: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if kind == "npy":
        with open(tmp, "wb") as f:
            np.save(f, obj)
    elif kind == "pt":
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def fold_rows(fs, k: int) -> tuple[np.ndarray, np.ndarray]:
    val = np.flatnonzero(fs.folds == k)
    tr = np.flatnonzero((fs.folds >= 0) & (fs.folds != k))
    assert not set(fs.slide_names[tr]) & set(fs.slide_names[val]), "slide overlap"
    assert len(val) and len(tr)
    return tr, val


def load_scaler(d: Path) -> Scaler:
    z = np.load(d / "scaler.npz")
    return Scaler(z["mean"], z["std"])


def train_one(fs, pre, variant: str, k: int, d: Path, epochs: int) -> None:
    if (d / "done.json").is_file():
        log(f"{variant}/fold{k}: done, skipped")
        return
    d.mkdir(parents=True, exist_ok=True)
    x, y = fs.features, fs.labels
    tr, val = fold_rows(fs, k)
    _atomic_save(val, d / "val_rows.npy", "npy")
    if (d / "scaler.npz").is_file():
        sc = load_scaler(d)
    else:
        sc = Scaler.fit(x, tr)
        np.savez(d / "scaler.npz", mean=sc.mean, std=sc.std)
    cw = balanced_weights(y[tr]) if variant == "b" else None
    cfg = TrainConfig(epochs=epochs, seed=k)

    done_epochs = sorted(int(p.stem.split("_")[1]) for p in d.glob("epoch_*.pt"))
    resume = None
    if done_epochs:
        resume = torch.load(d / f"epoch_{done_epochs[-1]:02d}.pt", weights_only=False)
        log(f"{variant}/fold{k}: resuming after epoch {resume['epoch'] + 1}")
    log(
        f"{variant}/fold{k}: train {len(tr)} tiles, val {len(val)} tiles, class weights {cw}"
    )
    t0 = time.time()
    for ep, model, state, loss in fit_epochs("linear", x, y, tr, sc, cfg, cw, resume):
        s = predict(model, x, val, sc)
        auroc, auprc = weighted_auroc_ap(y[val], s)
        _atomic_save(s, d / f"epoch_{ep:02d}_val.npy", "npy")
        _atomic_save(
            {
                "epoch": ep + 1,
                "train_loss": loss,
                "val_auroc": auroc,
                "val_auprc": auprc,
                "seconds": time.time() - t0,
            },
            d / f"epoch_{ep:02d}.json",
            "json",
        )
        _atomic_save(
            state, d / f"epoch_{ep:02d}.pt", "pt"
        )  # written last = epoch complete
        log(
            f"{variant}/fold{k} epoch {ep + 1}: loss {loss:.4f} val AUROC {auroc:.4f} AUPRC {auprc:.4f}"
        )

    hist = [json.loads((d / f"epoch_{e:02d}.json").read_text()) for e in range(epochs)]
    best = int(np.argmax([h["val_auprc"] for h in hist]))
    s_val = np.load(d / f"epoch_{best:02d}_val.npy")
    thr = youden_threshold(y[val], s_val)
    model = make_head("linear", x.shape[1], cfg.dropout)
    model.load_state_dict(
        torch.load(d / f"epoch_{best:02d}.pt", weights_only=False)["model"]
    )
    s_pre = predict(model, pre.features, np.arange(len(pre)), sc)
    _atomic_save(s_pre, d / "prelim_scores.npy", "npy")
    _atomic_save(
        {
            "variant": variant,
            "fold": k,
            "best_epoch": best + 1,
            "threshold": thr,
            "class_weights": cw,
            "epochs": epochs,
            "config": cfg.__dict__,
        },
        d / "done.json",
        "json",
    )
    log(f"{variant}/fold{k}: best epoch {best + 1}, Youden threshold {thr:.4f}")


def report(fs, pre, runs: Path, n_boot: int, variants, folds) -> dict:
    x, y = fs.features, fs.labels
    out: dict = {}
    for variant in variants:
        for k in folds:
            d = runs / variant / f"fold{k}"
            done = json.loads((d / "done.json").read_text())
            _, val = fold_rows(fs, k)
            assert np.array_equal(val, np.load(d / "val_rows.npy"))
            hist = [json.loads(p.read_text()) for p in sorted(d.glob("epoch_??.json"))]
            best = done["best_epoch"] - 1
            s_val = np.load(d / f"epoch_{best:02d}_val.npy")
            thr = done["threshold"]
            ev = evaluate(
                fs.slide_names[val], y[val], s_val, thr, n_boot=n_boot, seed=k
            )
            sc = load_scaler(d)
            null = []
            for seed in range(N_NULL):
                torch.manual_seed(seed)
                null.append(
                    weighted_auroc_ap(
                        y[val], predict(make_head("linear", x.shape[1]), x, val, sc)
                    )
                )
            null = np.array(null)
            s_pre = np.load(d / "prelim_scores.npy")
            ev_pre = evaluate(pre.slide_names, pre.labels, s_pre, thr, n_boot=0)
            out[f"{variant}/fold{k}"] = {
                "best_epoch": done["best_epoch"],
                "threshold": thr,
                "val": ev.point,
                "val_ci": ev.ci,
                "val_f1_at_youden": confusion(y[val], s_val, thr)["f1"],
                "null_auroc": (float(null[:, 0].mean()), float(null[:, 0].std())),
                "null_auprc": (float(null[:, 1].mean()), float(null[:, 1].std())),
                "epochs": hist,
                "prelim": ev_pre.point,
            }
            log(f"report {variant}/fold{k} done")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "report"])
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--prelim-cache", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--min-coverage", type=float, default=0.0)
    a = p.parse_args()

    #: label rules the runs assume; a cache built under other rules is refused
    config = LabelConfig(min_coverage=a.min_coverage)
    fs = load_features(a.train_cache, expected_config=config)
    pre = load_features(a.prelim_cache, expected_config=config)
    assert set(fs.splits) == {"train"} and set(pre.splits) == {"test_preliminary"}
    assert not set(fs.slide_names) & set(pre.slide_names)

    if a.cmd == "train":
        for variant in a.variants:
            for k in a.folds:
                train_one(fs, pre, variant, k, a.runs / variant / f"fold{k}", a.epochs)
    else:
        res = report(fs, pre, a.runs, a.n_boot, a.variants, a.folds)
        _atomic_save(res, a.runs / "report.json", "json")
        log(f"wrote {a.runs / 'report.json'}")


if __name__ == "__main__":
    main()
