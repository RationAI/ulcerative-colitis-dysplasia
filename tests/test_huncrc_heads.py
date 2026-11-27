"""Synthetic tests for ml.huncrc_heads."""

import numpy as np
import pytest
import torch

from ml.huncrc_head import fit_head
from ml.huncrc_heads import Scaler, binary_metrics, make_head, predict


def _data(n: int = 2000, d: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(np.int8)
    x = rng.normal(size=(n, d)).astype(np.float32) * 3 + 5
    x[:, 0] += y * 15.0  # separable along feature 0 (d' = 5)
    return x, y


def test_scaler_matches_numpy_on_subset() -> None:
    x, _ = _data()
    rows = np.arange(0, 2000, 3)[::-1]  # unsorted on purpose
    sc = Scaler.fit(x, rows, chunk=97)
    np.testing.assert_allclose(sc.mean, x[rows].mean(0), rtol=1e-5)
    np.testing.assert_allclose(sc.std, x[rows].std(0), rtol=1e-4)


def test_predict_keeps_row_order() -> None:
    x, _ = _data()
    sc = Scaler.fit(x, np.arange(len(x)))
    torch.manual_seed(0)
    model = make_head("linear", x.shape[1])
    rows = np.array([5, 3, 1999, 0, 3])
    s = predict(model, x, rows, sc, chunk=2)
    s_each = np.array([predict(model, x, np.array([r]), sc)[0] for r in rows])
    np.testing.assert_allclose(s, s_each, rtol=1e-6)


def test_heads_learn_separable_problem() -> None:
    x, y = _data()
    tr = np.arange(1500)
    va = np.arange(1500, 2000)
    sc = Scaler.fit(x, tr)
    for kind in ("linear", "mlp"):
        model, losses, _ = fit_head(kind, x, y, tr, sc, 15, lr=1e-3, batch_size=64)
        assert losses[-1] < losses[0]
        assert binary_metrics(y[va], predict(model, x, va, sc))["auroc"] > 0.95


def test_binary_metrics_constant_scores() -> None:
    y = np.array([1, 1, 1, 0])
    m = binary_metrics(y, np.ones(4))
    assert m["auroc"] == 0.5 and m["auprc"] == 0.75
    assert (m["precision"], m["recall"], m["specificity"]) == (0.75, 1.0, 0.0)
    assert abs(m["f1"] - 2 * 0.75 / 1.75) < 1e-12
    m0 = binary_metrics(y, np.zeros(4))
    assert m0["f1"] == 0.0 and m0["accuracy"] == 0.25


def test_balanced_weights_equalise_classes() -> None:
    from ml.huncrc_heads import balanced_weights

    y = np.array([1] * 70 + [0] * 30)
    wn, wp = balanced_weights(y)
    assert wn * 30 == pytest.approx(wp * 70)
    assert (wn * 30 + wp * 70) / 100 == pytest.approx(1.0)
