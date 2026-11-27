"""ml.huncrc_head: the torchmetrics monitoring stays out of the run files."""

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.huncrc_head import HunCRCHead


def test_monitoring_matches_sklearn_and_resets() -> None:
    torch.manual_seed(0)
    m = HunCRCHead("linear", in_features=8)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(300, 8)).astype(np.float32))
    y = torch.from_numpy((rng.random(300) < 0.4).astype(np.float32))
    m.train()
    scores = []
    for b in range(0, 300, 64):
        torch.manual_seed(b)  # the same dropout mask for the reference scores
        m.training_step((x[b : b + 64], y[b : b + 64], None))
        torch.manual_seed(b)
        scores.append(torch.sigmoid(m(x[b : b + 64])).detach())
    s = torch.cat(scores).numpy()
    mon = m.pop_monitoring()
    assert set(mon) == {"monitor_train_auroc", "monitor_train_auprc"}
    # float32 inside torchmetrics: close to sklearn, not bit-identical
    assert mon["monitor_train_auroc"] == pytest.approx(roc_auc_score(y, s), abs=1e-6)
    assert mon["monitor_train_auprc"] == pytest.approx(
        average_precision_score(y, s), abs=1e-6
    )
    m.training_step((x[:64], y[:64], None))
    assert m.pop_monitoring() != mon  # reset after each epoch


def test_monitoring_is_not_in_the_weights() -> None:
    m = HunCRCHead("mlp", in_features=8)
    assert all(k.startswith("head.") for k in m.state_dict())


def test_epoch_hook_logs_the_monitoring_series() -> None:
    from ml.huncrc_cv import epoch_hook

    calls: list[tuple[str, list[tuple[int, float]]]] = []

    class Logger:
        def log_series(self, name: str, points: list[tuple[int, float]]) -> None:
            calls.append((name, points))

    rec = {"epoch": 2, "train_loss": 0.5, "val_auroc": 0.9, "val_auprc": 0.8,
           "seconds": 1.0, "monitor_train_auroc": 0.7, "monitor_train_auprc": 0.6}  # fmt: skip
    epoch_hook(Logger())("C", 3, rec)  # type: ignore[arg-type]
    assert [n for n, _ in calls] == [
        "fold3_epoch_train_loss", "fold3_epoch_val_auroc", "fold3_epoch_val_auprc",
        "fold3_epoch_seconds", "fold3_epoch_monitor_train_auprc",
        "fold3_epoch_monitor_train_auroc",
    ]  # fmt: skip
    assert calls[-1][1] == [(2, 0.7)]
