"""Step 3b sanity checks on the cached train features (no MLflow, no test_final).

Validation = fold 0, training = folds 1-4, min_coverage = 0.

1. majority class            -> AUROC 0.5, AUPRC = prevalence
2. predict all dysplasia      -> precision = prevalence, recall 1
3. overfit 5 train slides     -> train AUROC > 0.99 (linear and MLP), else bug
4. shuffled training labels   -> validation AUROC ~ 0.5 on the REAL labels

    python scripts/huncrc/sanity_checks.py --cache <cache dir> --out <results.json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ml.data.huncrc_features import load_features
from ml.data.huncrc_labels import LabelConfig
from ml.huncrc_train import Scaler, TrainConfig, binary_metrics, predict, train_head


VAL_FOLD = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit-epochs", type=int, default=30)
    p.add_argument("--shuffled-epochs-linear", type=int, default=3)
    p.add_argument("--shuffled-epochs-mlp", type=int, default=2)
    a = p.parse_args()

    fs = load_features(a.cache, expected_config=LabelConfig(min_coverage=0.0))
    assert fs.meta["config"]["min_coverage"] == 0.0
    assert set(fs.splits) == {"train"}, set(fs.splits)
    x, y = fs.features, fs.labels
    val = np.flatnonzero(fs.folds == VAL_FOLD)
    tr = np.flatnonzero((fs.folds >= 1) & (fs.folds <= 4))
    assert len(val) + len(tr) == len(fs)
    tr_slides, val_slides = set(fs.slide_names[tr]), set(fs.slide_names[val])
    assert not tr_slides & val_slides, "train/validation slides overlap"
    log(
        f"train {len(tr)} tiles / {len(tr_slides)} slides, val {len(val)} tiles / {len(val_slides)} slides"
    )
    res: dict = {
        "split": {
            "train_tiles": len(tr),
            "train_slides": len(tr_slides),
            "val_tiles": len(val),
            "val_slides": len(val_slides),
            "train_prevalence": float(y[tr].mean()),
            "val_prevalence": float(y[val].mean()),
        }
    }

    # 1. majority class (from training labels), constant score
    majority = int(y[tr].mean() >= 0.5)
    const = np.full(len(val), float(majority), dtype=np.float32)
    m1 = binary_metrics(y[val], const)
    res["majority"] = {"majority_class": majority, **m1}
    log(f"majority={majority}: {m1}")

    # 2. predict all dysplasia
    res["all_dysplasia"] = binary_metrics(y[val], np.ones(len(val), dtype=np.float32))
    log(f"all dysplasia: {res['all_dysplasia']}")

    rng = np.random.default_rng(a.seed)

    # 3. overfit 5 train slides (3 positive-only sources + 2 negative sources), <=20k tiles each
    names_tr = fs.slide_names[tr]
    pos_slides = sorted({n for n in tr_slides if (y[tr][names_tr == n] == 1).any()})
    neg_slides = sorted({n for n in tr_slides if (y[tr][names_tr == n] == 0).any()})
    chosen = list(rng.choice(pos_slides, 3, replace=False)) + list(
        rng.choice(neg_slides, 2, replace=False)
    )
    of_rows = []
    for n in chosen:
        r = tr[names_tr == n]
        if len(r) > 20_000:
            r = np.sort(rng.choice(r, 20_000, replace=False))
        of_rows.append(r)
    of_rows = np.concatenate(of_rows)
    res["overfit"] = {
        "slides": [str(n) for n in chosen],
        "tiles": len(of_rows),
        "pos": int(y[of_rows].sum()),
        "neg": int((y[of_rows] == 0).sum()),
        "tiles_per_slide": {
            str(n): int((fs.slide_names[of_rows] == n).sum()) for n in chosen
        },
    }
    log(f"overfit set {res['overfit']}")
    sc = Scaler.fit(x, of_rows)
    for kind in ("linear", "mlp"):
        curve = []

        def track(ep, model, kind=kind, curve=curve):
            if ep in (0, 4, 9, 19) or ep == a.overfit_epochs - 1:
                au = binary_metrics(y[of_rows], predict(model, x, of_rows, sc))["auroc"]
                curve.append((ep + 1, au))
                log(f"overfit {kind} epoch {ep + 1}: train AUROC {au:.5f}")

        model, h = train_head(
            kind,
            x,
            y,
            of_rows,
            sc,
            TrainConfig(epochs=a.overfit_epochs, seed=a.seed),
            track,
        )
        m = binary_metrics(y[of_rows], predict(model, x, of_rows, sc))
        res["overfit"][kind] = {
            "train": m,
            "auroc_curve": curve,
            "loss": h.epoch_loss[-1],
            "epochs": a.overfit_epochs,
            "seconds": h.seconds,
        }
        log(
            f"overfit {kind}: train AUROC {m['auroc']:.5f} acc {m['accuracy']:.4f} ({h.seconds:.0f}s)"
        )

    # 4. shuffled labels within training folds, evaluate on REAL val labels
    y_shuf = y.copy()
    y_shuf[tr] = y[tr][np.random.default_rng(a.seed).permutation(len(tr))]
    assert (y_shuf[val] == y[val]).all() and y_shuf[tr].sum() == y[tr].sum()
    res["shuffled"] = {"frac_train_labels_changed": float((y_shuf[tr] != y[tr]).mean())}
    sc = Scaler.fit(x, tr)
    for kind, ep in (
        ("linear", a.shuffled_epochs_linear),
        ("mlp", a.shuffled_epochs_mlp),
    ):
        model, h = train_head(
            kind, x, y_shuf, tr, sc, TrainConfig(epochs=ep, seed=a.seed)
        )
        mv = binary_metrics(y[val], predict(model, x, val, sc))
        s_tr = predict(model, x, tr, sc)
        mt_shuf = binary_metrics(y_shuf[tr], s_tr)
        mt_real = binary_metrics(y[tr], s_tr)
        res["shuffled"][kind] = {
            "val_real": mv,
            "train_vs_shuffled": mt_shuf,
            "train_vs_real": mt_real,
            "epochs": ep,
            "loss": h.epoch_loss,
            "seconds": h.seconds,
        }
        log(
            f"shuffled {kind}: val AUROC {mv['auroc']:.4f}, train AUROC (shuffled) "
            f"{mt_shuf['auroc']:.4f}, train AUROC (real) {mt_real['auroc']:.4f} ({h.seconds:.0f}s)"
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=float))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
