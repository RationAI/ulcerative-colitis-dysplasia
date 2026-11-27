"""Smoke test: scripts/huncrc/linear_cv.py train (+ resume) and report."""

import json
import subprocess
from pathlib import Path

import numpy as np

from tests.huncrc_synth import build, run_script


def _train(
    caches: dict[str, Path], runs: Path, *extra: object
) -> subprocess.CompletedProcess[str]:
    return run_script(
        "linear_cv.py", "train",
        "--train-cache", caches["train"],
        "--prelim-cache", caches["test_preliminary"],
        "--runs", runs, "--epochs", 3, *extra,
    )  # fmt: skip


def test_linear_cv_train_resume_report(tmp_path: Path) -> None:
    caches = build(tmp_path)
    runs = tmp_path / "runs"
    _train(caches, runs)
    for v in ("a", "b"):
        for k in range(5):
            d = runs / v / f"fold{k}"
            done = json.loads((d / "done.json").read_text())
            assert 1 <= done["best_epoch"] <= 3
            assert 0.0 <= done["threshold"] <= 1.0
            assert len(list(d.glob("epoch_??.pt"))) == 3

    # simulate an interruption after epoch 2 of b/fold3, then resume
    d = runs / "b" / "fold3"
    before = np.load(d / "prelim_scores.npy")
    ep3 = np.load(d / "epoch_02_val.npy")
    for f in (
        "done.json",
        "epoch_02.pt",
        "epoch_02.json",
        "epoch_02_val.npy",
        "prelim_scores.npy",
    ):
        (d / f).unlink()
    proc = _train(caches, runs)
    assert "b/fold3: resuming after epoch 2" in proc.stdout
    assert "a/fold0: done, skipped" in proc.stdout
    np.testing.assert_array_equal(np.load(d / "epoch_02_val.npy"), ep3)
    np.testing.assert_array_equal(np.load(d / "prelim_scores.npy"), before)

    run_script(
        "linear_cv.py", "report",
        "--train-cache", caches["train"],
        "--prelim-cache", caches["test_preliminary"],
        "--runs", runs, "--n-boot", 20,
    )  # fmt: skip
    rep = json.loads((runs / "report.json").read_text())
    assert len(rep) == 10
    r = rep["a/fold0"]
    assert r["val"]["n_slides"] == 3 and r["prelim"]["n_slides"] == 4
    assert len(r["epochs"]) == 3
    assert set(r["val_ci"]) >= {"auroc", "auprc", "slide_auroc"}
    table = run_script("format_report.py", runs / "report.json").stdout
    assert "TEST_PRELIMINARY" in table and "mean" in table


def test_linear_cv_needs_matching_rules(tmp_path: Path) -> None:
    caches = build(tmp_path)
    proc_args = (caches, tmp_path / "runs", "--min-coverage", "0.5")
    try:
        _train(*proc_args)
    except AssertionError as e:
        assert "StaleCacheError" in str(e)
    else:
        raise AssertionError("a min_coverage=0.5 run accepted a min_coverage=0 cache")
