"""Synthetic tests for ml.huncrc_eval."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.huncrc_eval import (
    confusion,
    evaluate,
    slide_scores,
    weighted_auroc_ap,
    youden_threshold,
)


def _data(seed: int = 0, n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.6).astype(np.int8)
    score = np.round(rng.random(n) * 0.6 + y * 0.3, 2)  # rounding -> many ties
    return y, score


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_metrics_match_sklearn(seed: int) -> None:
    y, s = _data(seed)
    w = np.random.default_rng(seed).integers(0, 4, len(y)).astype(float)
    a, ap = weighted_auroc_ap(y, s, w)
    assert a == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)
    assert ap == pytest.approx(
        average_precision_score(y, s, sample_weight=w), abs=1e-12
    )
    a1, ap1 = weighted_auroc_ap(y, s)
    assert a1 == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert ap1 == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_youden_threshold_maximises_j() -> None:
    y, s = _data()
    t = youden_threshold(y, s)
    c = confusion(y, s, t)
    j = c["sensitivity"] + c["specificity"] - 1
    for u in np.unique(s):
        cu = confusion(y, s, u)
        assert cu["sensitivity"] + cu["specificity"] - 1 <= j + 1e-12


def test_youden_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    t = youden_threshold(y, s)
    assert 0.2 < t <= 0.8
    assert confusion(y, s, t)["sensitivity"] == 1.0 == confusion(y, s, t)["specificity"]


def test_slide_scores_and_purity() -> None:
    names = np.array(["a", "a", "b", "b", "c"])
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.2, 0.4, 0.1, 0.3, 0.9])
    u, sy, sm = slide_scores(names, y, s)
    assert u.tolist() == ["a", "b", "c"] and sy.tolist() == [1, 0, 1]
    np.testing.assert_allclose(sm, [0.3, 0.2, 0.9])
    with pytest.raises(AssertionError):
        slide_scores(names, np.array([1, 0, 0, 0, 1]), s)


def test_evaluate_point_matches_sklearn_and_ci_brackets() -> None:
    rng = np.random.default_rng(0)
    slides = np.repeat([f"s{i:02d}" for i in range(20)], 50)
    slide_y = np.repeat((np.arange(20) < 12).astype(np.int8), 50)
    s = np.clip(rng.normal(0.4 + 0.2 * slide_y, 0.2), 0, 1)
    ev = evaluate(slides, slide_y, s, threshold=0.5, n_boot=200)
    assert ev.point["auroc"] == pytest.approx(roc_auc_score(slide_y, s), abs=1e-12)
    assert ev.point["auprc"] == pytest.approx(
        average_precision_score(slide_y, s), abs=1e-12
    )
    c = confusion(slide_y, s, 0.5)
    for k in ("sensitivity", "specificity", "f1"):
        assert ev.point[k] == pytest.approx(c[k])
    for k, (lo, hi) in ev.ci.items():
        assert lo <= ev.point[k] + 1e-9 and ev.point[k] - 1e-9 <= hi, k
    assert ev.point["n_pos_slides"] == 12 and ev.point["n_neg_slides"] == 8


def test_zero_weight_top_scores_do_not_give_nan() -> None:
    y, s = _data()
    w = np.ones(len(y))
    w[np.argsort(-s)[:200]] = 0.0  # highest-scored tiles absent from the resample
    a, ap = weighted_auroc_ap(y, s, w)
    assert np.isfinite(a) and np.isfinite(ap)
    assert ap == pytest.approx(
        average_precision_score(y, s, sample_weight=w), abs=1e-12
    )
    assert a == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)


def test_slide_bootstrap_point_and_degenerate_ci() -> None:
    from ml.huncrc_eval import slide_bootstrap

    # 4 slides, 3 tiles each; every slide has the same fraction -> CI collapses
    item_slide = np.repeat(np.arange(4), 3)
    hit = np.tile([1.0, 0.0, 1.0], 4)
    p, lo, hi = slide_bootstrap(
        item_slide, np.zeros(4), lambda w: (w * hit).sum() / w.sum(), 200
    )
    assert (
        p == pytest.approx(2 / 3)
        and lo == pytest.approx(2 / 3)
        and hi == pytest.approx(2 / 3)
    )


def test_slide_bootstrap_keeps_strata_sizes() -> None:
    from ml.huncrc_eval import slide_bootstrap

    strata = np.array([0, 0, 0, 1, 1])
    seen = []

    def stat(w: np.ndarray) -> float:
        seen.append((w[:3].sum(), w[3:].sum()))
        return 0.0

    slide_bootstrap(np.arange(5), strata, stat, n_boot=50)
    assert all(a == 3 and b == 2 for a, b in seen)


def test_slide_bootstrap_ci_brackets_point() -> None:
    from ml.huncrc_eval import slide_bootstrap

    rng = np.random.default_rng(0)
    item_slide = np.repeat(np.arange(10), 20)
    hit = (rng.random(200) < np.repeat(rng.random(10), 20)).astype(float)
    p, lo, hi = slide_bootstrap(
        item_slide, np.zeros(10), lambda w: (w * hit).sum() / w.sum(), 500
    )
    assert lo < p < hi


def test_evaluate_with_mixed_slides() -> None:
    from ml.huncrc_eval import evaluate

    rng = np.random.default_rng(1)
    names = np.repeat([f"s{i}" for i in range(8)], 40)
    # slides 0-4 mix positives and negatives (within-slide negatives), 5-7 negative only
    y = np.where(np.arange(len(names)) < 200, rng.random(len(names)) < 0.6, 0).astype(
        np.int8
    )
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    ev = evaluate(names, y, s, threshold=0.5, n_boot=100)
    assert ev.point["pure_slides"] is False
    assert np.isnan(ev.point["slide_auroc"])
    assert ev.point["auroc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert (ev.point["n_pos_slides"], ev.point["n_neg_slides"]) == (5, 3)
