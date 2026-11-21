"""Render linear_cv.py's report.json as a plain-text table (for thesis.txt).

python scripts/huncrc/format_report.py R/report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


NAMES = {"a": "(a) unweighted", "b": "(b) class-balanced"}


def ci(r: dict, k: str) -> str:
    lo, hi = r["val_ci"][k]
    return f"{r['val'][k]:.3f} [{lo:.3f}-{hi:.3f}]"


def main(path: str) -> None:
    rep = json.loads(Path(path).read_text())
    variants = sorted({k.split("/")[0] for k in rep})
    lines = []
    for v in variants:
        rows = {k: r for k, r in rep.items() if k.startswith(f"{v}/")}
        lines.append(
            f"\n{NAMES.get(v, v)} -- VALIDATION fold k (model trained on the other 4 folds)"
        )
        lines.append(
            "fold  ep  thr     tiles   pos/neg sl  prev   AUROC [95% CI]       AUPRC [95% CI]       "
            "sens   spec   F1     slide AUROC [95% CI]  null AUROC    null AUPRC"
        )
        for k, r in sorted(rows.items()):
            p = r["val"]
            lines.append(
                f"{k.split('fold')[1]:>4}  {r['best_epoch']:>2}  {r['threshold']:.4f}  "
                f"{p['n_tiles']:>7}  {p['n_pos_slides']:>2}/{p['n_neg_slides']:<2}     "
                f"{p['prevalence']:.3f}  {ci(r, 'auroc')}  {ci(r, 'auprc')}  "
                f"{p['sensitivity']:.3f}  {p['specificity']:.3f}  {p['f1']:.3f}  "
                f"{ci(r, 'slide_auroc')}   "
                f"{r['null_auroc'][0]:.3f}+/-{r['null_auroc'][1]:.3f}  "
                f"{r['null_auprc'][0]:.3f}+/-{r['null_auprc'][1]:.3f}"
            )
        lines.append(
            f"\n{NAMES.get(v, v)} -- TEST_PRELIMINARY (each fold model at its own validation threshold)"
        )
        lines.append("fold  AUROC  AUPRC  sens   spec   F1     slide AUROC")
        keys = ["auroc", "auprc", "sensitivity", "specificity", "f1", "slide_auroc"]
        vals = []
        for k, r in sorted(rows.items()):
            p = r["prelim"]
            vals.append([p[m] for m in keys])
            lines.append(
                f"{k.split('fold')[1]:>4}  " + "  ".join(f"{p[m]:.3f}" for m in keys)
            )
        a = np.array(vals)
        lines.append(
            "mean  "
            + "  ".join(f"{m:.3f}" for m in a.mean(0))
            + "\n  sd  "
            + "  ".join(f"{s:.3f}" for s in a.std(0, ddof=1))
        )
        p0 = next(iter(rows.values()))["prelim"]
        lines.append(
            f"(test_preliminary: {p0['n_tiles']} tiles, {p0['n_pos_slides']} pos / "
            f"{p0['n_neg_slides']} neg slides, prevalence {p0['prevalence']:.3f})"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
