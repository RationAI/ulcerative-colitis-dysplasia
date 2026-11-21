"""Synthetic tests for ml.huncrc_train."""

import numpy as np
import pytest

from ml.huncrc_train import Scaler, TrainConfig, binary_metrics, predict, train_head


def _data(n=2000, d=16, seed=0):
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
    x, y = _data()
    sc = Scaler.fit(x, np.arange(len(x)))
    model, _ = train_head(
        "linear", x, y, np.arange(1000), sc, TrainConfig(epochs=1, super_batch=300)
    )
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
        cfg = TrainConfig(epochs=15, lr=1e-3, batch_size=64, super_batch=500)
        model, h = train_head(kind, x, y, tr, sc, cfg)
        assert h.epoch_loss[-1] < h.epoch_loss[0]
        assert binary_metrics(y[va], predict(model, x, va, sc))["auroc"] > 0.95


def test_binary_metrics_constant_scores() -> None:
    y = np.array([1, 1, 1, 0])
    m = binary_metrics(y, np.ones(4))
    assert m["auroc"] == 0.5 and m["auprc"] == 0.75
    assert (m["precision"], m["recall"], m["specificity"]) == (0.75, 1.0, 0.0)
    assert abs(m["f1"] - 2 * 0.75 / 1.75) < 1e-12
    m0 = binary_metrics(y, np.zeros(4))
    assert m0["f1"] == 0.0 and m0["accuracy"] == 0.25


def test_fit_epochs_resume_is_exact() -> None:
    import torch

    from ml.huncrc_train import fit_epochs

    x, y = _data()
    rows = np.arange(1500)
    sc = Scaler.fit(x, rows)
    cfg = TrainConfig(epochs=4, batch_size=64, super_batch=500, lr=1e-3)
    full = list(fit_epochs("mlp", x, y, rows, sc, cfg))
    saved = torch.load(_roundtrip(full[1][2]), weights_only=False)
    resumed = list(fit_epochs("mlp", x, y, rows, sc, cfg, resume=saved))
    assert [e for e, *_ in resumed] == [2, 3]
    for a, b in zip(full[2:], resumed, strict=True):
        assert a[3] == pytest.approx(b[3], rel=1e-6)
        for k in a[2]["model"]:
            torch.testing.assert_close(a[2]["model"][k], b[2]["model"][k])


def _roundtrip(obj):
    import io

    import torch

    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return buf


def test_balanced_weights_equalise_classes() -> None:
    from ml.huncrc_train import balanced_weights

    y = np.array([1] * 70 + [0] * 30)
    wn, wp = balanced_weights(y)
    assert wn * 30 == pytest.approx(wp * 70)
    assert (wn * 30 + wp * 70) / 100 == pytest.approx(1.0)
