"""Steps 8-9: linear / MLP head with within-slide negatives, 5-fold CV + Step-7 eval.

Labels (cache built with ``build_feature_cache.py --within-slide-negatives
--exclude-carcinoma``): positives = dysplasia tiles; negatives = safe-slide
negatives + strict-normal tiles of positive slides; carcinoma tiles and the
excluded slides never train. Negatives are capped per slide in TRAINING only
(validation keeps every tile).

    # resumable: finished (variant, fold) runs are skipped, an interrupted run
    # resumes after its last complete epoch
    python scripts/huncrc/within_slide_cv.py train --train-cache C --prelim-all-cache P --runs R
    # report (works on partial results), next to the saved Step-4a models (A);
    # --other-runs adds variants trained elsewhere (e.g. Step-8 B/C next to D/E)
    python scripts/huncrc/within_slide_cv.py report --train-cache C --prelim-all-cache P \
        --runs R [--other-runs R2 ...] [--a-runs <step4a linear_cv dir>] \
        --annotations A --tiling T --slide-csv CSV

Variants (all on the new labels): B = linear, unweighted BCE; C = linear,
class-balanced BCE; D = MLP 1536-512-256-1, unweighted; E = MLP, class-balanced.
A-unw / A-bal = the Step-4a variants (a) / (b), reused without retraining.
Same protocol as Step 4a: scaler from the training folds only, up to N epochs,
best epoch = max validation AUPRC, Youden threshold on the validation fold,
seed = fold index. test_final is never read; no MLflow.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from rationai.mlkit import Trainer

from ml.atomic_checkpoint_io import AtomicCheckpointIO
from ml.callbacks import YoudenThreshold
from ml.data.huncrc_data_module import HunCRCFoldData
from ml.data.huncrc_features import FeatureSet, load_features
from ml.data.huncrc_folds import neg_source
from ml.data.huncrc_labels import (
    LabelConfig,
    load_slide_table,
    rules_diff,
    rules_fingerprint,
    safe_slide_names,
)
from ml.huncrc_eval import evaluate, weighted_auroc_ap
from ml.huncrc_head import HunCRCHead
from ml.huncrc_train import (
    Scaler,
    TrainConfig,
    make_head,
    predict,
)
from ml.huncrc_wsmetrics import prelim_masks, prelim_report
from ml.run_files import save


VARIANTS = {  # name -> (head, class weighting)
    "B": ("linear", None),
    "C": ("linear", "balanced"),
    "D": ("mlp", None),
    "E": ("mlp", "balanced"),
}
DEFAULT_TRAIN_VARIANTS = ["B", "C"]  # Step 8; Step 9 passes --variants D E
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


EpochHook = Callable[[str, int, dict[str, Any]], None]


def load_caches(a) -> tuple[FeatureSet, FeatureSet]:
    """Train and test_preliminary caches; the second must hold the same embedding."""
    fs = load_features(a.train_cache, expected_config=a.rules)
    model, dim = fs.embedding
    pre = load_features(
        a.prelim_all_cache,
        expected_config=OLD_RULES,
        mode="all",
        expected_model=model,
        expected_dim=dim,
    )
    log(f"embedding: {model}, {dim}-d (train and test_preliminary caches)")
    return fs, pre


def train_rules(a) -> LabelConfig:
    """The Step-8 rules, or (``--label-rules-from-cache``) those of the train cache.

    Either way ``load_features`` then checks the cache fingerprint against them.
    """
    if not a.label_rules_from_cache:
        return NEW_RULES
    meta = json.loads((a.train_cache / "meta.json").read_text())
    rules = LabelConfig(**meta["config"])
    diff = rules_diff(rules, NEW_RULES)
    if diff:
        log(
            "WARNING: label rules differ from the Step-8 rules (now, Step 8): "
            + "; ".join(f"{k}={v[0]} (Step 8: {v[1]})" for k, v in diff.items())
        )
    return rules


class EpochFiles(Callback):
    """The reference loop's per-epoch outputs.

    Validation scores and metrics, then the weights, written last, so a
    complete epoch_NN.pt means a complete epoch.
    """

    def __init__(self, d: Path, name: str, epochs: int,
                 on_epoch: EpochHook | None) -> None:  # fmt: skip
        self.d, self.name, self.epochs, self.on_epoch = d, name, epochs, on_epoch
        self.t0 = time.time()

    def on_train_epoch_end(self, trainer: Trainer, module: HunCRCHead) -> None:
        ep, d = trainer.current_epoch, self.d
        s = module.val_scores  # Lightning's validation loop ran just before
        auroc, auprc = weighted_auroc_ap(trainer.datamodule.val_labels, s)
        save(s, d / f"epoch_{ep:02d}_val.npy", "npy")
        save(
            {
                "epoch": ep + 1,
                "train_loss": module.train_loss,
                "val_auroc": auroc,
                "val_auprc": auprc,
                "seconds": time.time() - self.t0,
            },
            d / f"epoch_{ep:02d}.json",
        )
        #: deep copy: state_dict() returns live references that later epochs mutate
        state = copy.deepcopy({"epoch": ep, "model": module.head.state_dict(),
                               "optimizer": trainer.optimizers[0].state_dict()})  # fmt: skip
        save(state, d / f"epoch_{ep:02d}.pt", "pt")  # written last = epoch complete
        # monitoring only, never written to a run file
        monitor = module.pop_monitoring()
        # e.g. live MLflow from ml/huncrc_cv.py; the hook must not raise
        if self.on_epoch is not None:
            variant, k = self.name.split("/fold")
            rec = json.loads((d / f"epoch_{ep:02d}.json").read_text())
            self.on_epoch(variant, int(k), rec | monitor)
        log(
            f"{self.name} epoch {ep + 1}/{self.epochs}: loss {module.train_loss:.4f} val AUROC {auroc:.4f} AUPRC {auprc:.4f} ({time.time() - self.t0:.0f}s)"
        )


def train_one(
    fs, pre, variant, k, d: Path, a, on_epoch: EpochHook | None = None
) -> None:
    if (d / "done.json").is_file():
        log(f"{variant}/fold{k}: done, skipped")
        return
    head, weighting = VARIANTS[variant]
    cfg = TrainConfig(epochs=a.epochs, seed=k)
    data = HunCRCFoldData(
        fs, k, d, safe_slide_names(a.slide_table, a.rules), weighting, a.neg_cap,
        a.cap_seed, a.smoke_tiles_per_slide, cfg.batch_size, cfg.super_batch, cfg.seed,
    )  # fmt: skip
    data.setup("fit")  # rows, scaler and class weights, for the log line below
    x, sc, cnt, cw = fs.features, data.scaler, data.counts, data.class_weights
    ckpt = d / "checkpoints" / "last.ckpt"  # Lightning's, after every epoch
    # its "epoch" is the number of completed epochs
    resume = torch.load(ckpt, weights_only=False)["epoch"] if ckpt.is_file() else None
    log(
        f"{variant}/fold{k}: start ({'resume after epoch ' + str(resume) if resume is not None else 'new'}); "
        f"head {head}, train {cnt['train']}, val {cnt['val']}, class weights {cw}"
    )
    # the reference loop read 65536-row blocks and cut them into batches; with a
    # whole number of batches per block that is plain consecutive batching
    assert cfg.super_batch % cfg.batch_size == 0
    torch.manual_seed(cfg.seed)  # head initialisation, as in the reference
    module = HunCRCHead(
        head, x.shape[1], cfg.dropout, cfg.lr, cfg.weight_decay, cfg.seed
    )
    youden = YoudenThreshold(d)
    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[
            EpochFiles(d, f"{variant}/fold{k}", cfg.epochs, on_epoch),
            youden,
            # after the run files (checkpoint callbacks run last), as the reference
            # wrote the weights last; by default it would save at validation end
            ModelCheckpoint(
                dirpath=d / "checkpoints",
                save_top_k=0,
                save_last=True,
                save_on_train_epoch_end=True,
            ),
        ],
        plugins=[AtomicCheckpointIO()],
    )
    t0 = time.time()
    trainer.fit(module, datamodule=data, ckpt_path=ckpt if resume is not None else None)
    best, thr = youden.best_epoch, youden.threshold
    model = make_head(head, x.shape[1], cfg.dropout)
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
            "head": head,
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


def _model(d: Path, dim: int, head: str) -> tuple[torch.nn.Module, Scaler, dict]:
    done = json.loads((d / "done.json").read_text())
    head = done.get("head", head)  # Step-8 / Step-4a runs predate the field
    z = np.load(d / "scaler.npz")
    m = make_head(head, dim)
    m.load_state_dict(
        torch.load(d / f"epoch_{done['best_epoch'] - 1:02d}.pt", weights_only=False)[
            "model"
        ]
    )
    return m, Scaler(z["mean"], z["std"]), done


def report(a) -> dict:
    fs, pre = load_caches(a)
    safe = safe_slide_names(a.slide_table, a.rules)
    src = neg_source(fs, safe)
    old_label = (fs.labels == 1) | (src == "safe_slide")  # Step-4a label set
    masks = prelim_masks(pre, a.slide_table, a.annotations)
    sets: dict[str, tuple[Path, str]] = {}
    if a.a_runs:
        sets = {name: (a.a_runs / v, "linear") for name, v in A_VARIANTS.items()}
    else:
        log("report: no --a-runs, Step-4a models (A-unw, A-bal) skipped")
    for v, (head, _) in VARIANTS.items():
        roots = [r / v for r in [a.runs, *a.other_runs] if (r / v).is_dir()]
        if len(roots) > 1:
            raise SystemExit(f"variant {v} found in several runs dirs: {roots}")
        if roots:
            sets[v] = (roots[0], head)
    out: dict[str, Any] = {"model_sets": {}, "counts": {}, "sources": {}}
    for name, (root, head) in sets.items():
        folds = [k for k in range(5) if (root / f"fold{k}" / "done.json").is_file()]
        if not folds:
            log(f"report: {name}: no finished folds, skipped")
            continue
        R: dict[str, Any] = {"folds": folds, "val": {}, "head": head}
        out["sources"][name] = str(root)
        prelim_scores, thrs = [], []
        for k in folds:
            d = root / f"fold{k}"
            model, sc, done = _model(d, fs.features.shape[1], head)
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
                            make_head(done.get("head", head), fs.features.shape[1]),
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
    L.append("\nmodel sets (head, runs dir)")
    for n in names:
        L.append(
            f"{n:<7} {rep['model_sets'][n].get('head', '?'):<7} {rep.get('sources', {}).get(n, '?')}"
        )
    return "\n".join(L)


def main(argv: list[str] | None = None, on_epoch: EpochHook | None = None) -> None:
    """CLI; ``argv``/``on_epoch`` let ml/huncrc_cv.py (Hydra) build the same call."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["train", "report"])
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--prelim-all-cache", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--tiling", type=Path, required=True)
    p.add_argument("--slide-csv", type=Path, required=True)
    p.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_TRAIN_VARIANTS,
        choices=list(VARIANTS),
        help="train: B/C linear, D/E MLP (default: B C)",
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
        "--a-runs",
        type=Path,
        help="report: Step-4a linear_cv runs dir (variants a, b); omit to skip A",
    )
    p.add_argument(
        "--other-runs",
        type=Path,
        nargs="*",
        default=[],
        help="report: more runs dirs holding B/C/D/E variants (e.g. Step 8)",
    )
    p.add_argument(
        "--annotations", type=Path, help="report: dir of <slide>.json squares"
    )
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, help="report: output dir (default: --runs)")
    p.add_argument(
        "--label-rules-from-cache",
        action="store_true",
        help="train on the label rules the train cache was built with "
        "(default: the Step-8 rules only)",
    )
    a = p.parse_args(argv)
    a.rules = train_rules(a)
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
        fs, pre = load_caches(a)
        assert set(fs.splits) == {"train"} and set(pre.splits) == {"test_preliminary"}
        a.runs.mkdir(parents=True, exist_ok=True)
        (a.runs / "run_config.json").write_text(
            json.dumps(
                {
                    **{
                        k: str(v)
                        for k, v in vars(a).items()
                        if k not in {"slide_table", "rules"}
                    },
                    "label_rules_fingerprint": rules_fingerprint(a.rules),
                    "embedding_model": fs.embedding[0],
                    "embedding_dim": fs.embedding[1],
                },
                indent=2,
            )
        )
        for v in a.variants:
            for k in a.folds:
                train_one(fs, pre, v, k, a.runs / v / f"fold{k}", a, on_epoch)
        log("train: all requested runs finished")
    else:
        if not a.annotations:
            p.error("report needs --annotations")
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
