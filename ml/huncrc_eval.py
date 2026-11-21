"""Evaluation helpers: threshold choice, slide aggregation, slide bootstrap.

Weighted AUROC / AP are computed from one score sort, so a bootstrap over
slides (which only changes per-tile multiplicities) costs a few cumsums per
resample. They agree with sklearn's ``roc_auc_score`` /
``average_precision_score`` with ``sample_weight`` (see the tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_curve


if TYPE_CHECKING:
    from collections.abc import Callable


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (predict pos if score >= t)."""
    fpr, tpr, thr = roc_curve(y, score)
    j = tpr - fpr
    i = int(np.argmax(j))
    t = float(thr[i])
    return min(t, 1.0)  # roc_curve's first threshold is +inf


def confusion(
    y: np.ndarray, score: np.ndarray, t: float, w: np.ndarray | None = None
) -> dict[str, float]:
    w = np.ones(len(y)) if w is None else w
    pred = score >= t
    pos = y == 1
    tp = float(w[pred & pos].sum())
    fp = float(w[pred & ~pos].sum())
    fn = float(w[~pred & pos].sum())
    tn = float(w[~pred & ~pos].sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}


class _Sorted:
    """Scores sorted descending with tie-group ends, for repeated weighted metrics."""

    def __init__(self, y: np.ndarray, score: np.ndarray) -> None:
        order = np.argsort(-score, kind="mergesort")
        self.order = order
        self.y = (y[order] == 1).astype(np.float64)
        s = score[order]
        # last index of every group of equal scores
        self.ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])

    def metrics(self, w_sorted: np.ndarray) -> tuple[float, float]:
        tps = np.cumsum(w_sorted * self.y)[self.ends]
        fps = np.cumsum(w_sorted * (1 - self.y))[self.ends]
        P, N = tps[-1], fps[-1]
        if P <= 0 or N <= 0:
            return float("nan"), float("nan")
        tpr = np.r_[0.0, tps / P]
        fpr = np.r_[0.0, fps / N]
        auroc = float(np.trapezoid(tpr, fpr))
        denom = tps + fps
        #: zero-weight leading groups (bootstrap) have no recall step; count them as 0
        prec = np.divide(tps, denom, out=np.zeros_like(tps), where=denom > 0)
        recall = tps / P
        ap = float(np.sum(np.diff(np.r_[0.0, recall]) * prec))
        return auroc, ap


#: public name for repeated weighted metrics on one fixed score ordering
SortedScores = _Sorted


def weighted_auroc_ap(
    y: np.ndarray, score: np.ndarray, w: np.ndarray | None = None
) -> tuple[float, float]:
    s = _Sorted(y, score)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    return s.metrics(w[s.order])


def slide_scores(
    names: np.ndarray, y: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean tile score per slide. Slide label = 1 if it has any positive labelled tile.

    Under the Step 2 rules every slide is pure (only positives or only
    negatives among its labelled tiles); asserted here.
    """
    uniq, inv = np.unique(names, return_inverse=True)
    cnt = np.bincount(inv)
    mean = np.bincount(inv, weights=score) / cnt
    pos = np.bincount(inv, weights=(y == 1)) / cnt
    assert np.all((pos == 0) | (pos == 1)), "a slide mixes positive and negative tiles"
    return uniq, (pos == 1).astype(np.int8), mean


@dataclass
class FoldEval:
    point: dict[str, float]
    ci: dict[str, tuple[float, float]]
    n_boot: int


def evaluate(
    names: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> FoldEval:
    """Point estimates + 95% percentile CI from a bootstrap over SLIDES.

    Slides are resampled with replacement within their slide class (slides with
    / without positive tiles separately), so every resample has both classes.
    The threshold is held fixed. ``slide_auroc`` (mean tile score per slide) is
    only defined when every slide is pure (only positive or only negative
    labelled tiles); with within-slide negatives it is NaN.
    """
    uniq, inv = np.unique(names, return_inverse=True)
    pos_frac = np.bincount(inv, weights=(y == 1)) / np.bincount(inv)
    pure = bool(np.all((pos_frac == 0) | (pos_frac == 1)))
    s_y = (pos_frac > 0).astype(np.int8)
    if pure:
        s_names, s_y_pure, s_score = slide_scores(names, y, score)
        assert (s_names == uniq).all() and (s_y_pure == s_y).all()
        s_srt = _Sorted(s_y, s_score)
    srt = _Sorted(y, score)
    inv_sorted = inv[srt.order]
    pred_sorted = score[srt.order] >= threshold
    y_sorted = srt.y

    def at(slide_w: np.ndarray) -> dict[str, float]:
        w = slide_w[inv_sorted]
        auroc, ap = srt.metrics(w)
        tp = float(w[pred_sorted & (y_sorted == 1)].sum())
        fn = float(w[~pred_sorted & (y_sorted == 1)].sum())
        fp = float(w[pred_sorted & (y_sorted == 0)].sum())
        tn = float(w[~pred_sorted & (y_sorted == 0)].sum())
        s_auroc = s_srt.metrics(slide_w[s_srt.order])[0] if pure else float("nan")
        return {
            "auroc": auroc,
            "auprc": ap,
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "slide_auroc": s_auroc,
            "prevalence": (tp + fn) / (tp + fn + fp + tn),
        }

    point = at(np.ones(len(uniq)))
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(s_y == 1)
    neg_idx = np.flatnonzero(s_y == 0)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        c = np.zeros(len(uniq))
        np.add.at(c, rng.choice(pos_idx, len(pos_idx)), 1)
        np.add.at(c, rng.choice(neg_idx, len(neg_idx)), 1)
        for k, v in at(c).items():
            draws[k].append(v)
    ci = (
        {
            k: (
                (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
                if np.isfinite(v).any()
                else (float("nan"), float("nan"))
            )
            for k, v in draws.items()
        }
        if n_boot
        else {}
    )
    point.update(
        n_tiles=len(y),
        n_slides=len(uniq),
        n_pos_slides=len(pos_idx),
        n_neg_slides=len(neg_idx),
        pure_slides=pure,
    )
    return FoldEval(point=point, ci=ci, n_boot=n_boot)


def slide_bootstrap(
    item_slide: np.ndarray,
    slide_strata: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and 95% percentile CI of ``stat`` under a slide bootstrap.

    ``item_slide[i]`` is the slide index (0..S-1) of item ``i`` (a tile, or a
    slide itself for slide-level statistics); ``slide_strata[s]`` is the
    stratum of slide ``s``. Slides are resampled with replacement within each
    stratum, and ``stat(w)`` is called with per-item weights ``w`` (the
    multiplicity of the item's slide). Returns ``(point, lo, hi)``.
    """
    item_slide = np.asarray(item_slide)
    slide_strata = np.asarray(slide_strata)
    n_slides = len(slide_strata)
    point = float(stat(np.ones(len(item_slide))))
    if n_boot == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(slide_strata == g) for g in np.unique(slide_strata)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        c = np.zeros(n_slides)
        for g in groups:
            np.add.at(c, rng.choice(g, len(g)), 1)
        draws[b] = stat(c[item_slide])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)
