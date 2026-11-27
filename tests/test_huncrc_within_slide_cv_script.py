"""Smoke test: scripts/huncrc/within_slide_cv.py train (+ resume) and report."""

import json
import shutil
from pathlib import Path

import numpy as np

from tests.huncrc_synth import build, run_script, write_step4a_runs


def test_within_slide_cv_end_to_end(tmp_path: Path) -> None:
    c = build(tmp_path)
    paths = ["--tiling", c["tiling"], "--slide-csv", c["slide_csv"]]
    # Step-4a-shaped runs as the "A" reference (labelled caches + all-tile scores)
    a_runs = tmp_path / "a_runs"
    write_step4a_runs(a_runs, c["train"], c["test_preliminary"])
    run_script(
        "prelim_breakdown.py", "score", "--runs", a_runs,
        "--all-cache", c["test_preliminary_all"], "--prelim-cache", c["test_preliminary"],
    )  # fmt: skip
    runs = tmp_path / "ws_runs"
    common = [
        "--train-cache", c["train_within_slide"],
        "--prelim-all-cache", c["test_preliminary_all"], "--runs", runs, *paths,
    ]  # fmt: skip
    proc = run_script(
        "within_slide_cv.py", "train", *common, "--epochs", 2, "--neg-cap", 20
    )
    assert proc.stdout.count("DONE") == 10
    cnt = json.loads((runs / "B" / "fold0" / "counts.json").read_text())
    # cap 20 per slide; training folds 1-4 hold 4 safe slides (60 negatives each)
    # and 8 adenoma slides (28 strict-normal tiles each)
    assert cnt["train"]["neg_safe_slide"] == 4 * 20
    assert cnt["train"]["neg_positive_slide_normal"] == 8 * 20
    assert cnt["train_before_cap"]["neg_positive_slide_normal"] == 8 * 28
    assert cnt["val"]["neg_positive_slide_normal"] == 2 * 28  # validation is not capped

    # interrupt C/fold2 after epoch 1, resume -> identical scores. Training
    # resumes from Lightning's last.ckpt, so the interruption is a run that
    # stopped after epoch 1 and never finished the fold (no done.json)
    d = runs / "C" / "fold2"
    before = np.load(d / "prelim_all_scores.npy")
    shutil.rmtree(d)
    run_script("within_slide_cv.py", "train", *common, "--epochs", 1, "--neg-cap", 20)
    for f in ("done.json", "prelim_all_scores.npy"):
        (d / f).unlink()
    proc = run_script(
        "within_slide_cv.py", "train", *common, "--epochs", 2, "--neg-cap", 20
    )
    assert "C/fold2: start (resume after epoch 1)" in proc.stdout
    assert proc.stdout.count("done, skipped") == 9
    np.testing.assert_array_equal(np.load(d / "prelim_all_scores.npy"), before)

    # report on partial results: remove one B fold
    (runs / "B" / "fold4" / "done.json").unlink()
    run_script(
        "within_slide_cv.py", "report", *common, "--a-runs", a_runs,
        "--annotations", c["annotations"], "--n-boot", 10,
    )  # fmt: skip
    rep = json.loads((runs / "report.json").read_text())
    assert set(rep["model_sets"]) == {"A-unw", "A-bal", "B", "C"}
    assert rep["model_sets"]["B"]["folds"] == [0, 1, 2, 3]
    v = rep["model_sets"]["B"]["val"]["fold0"]
    assert np.isnan(v["new"]["slide_auroc"]) and not v["new"]["pure_slides"]
    assert v["old"]["pure_slides"]  # the Step-4a label set has pure slides
    s = rep["model_sets"]["C"]["prelim"]["summary"]
    assert 0.0 <= s["ws_auroc"]["mean"] <= 1.0 and "ci" in s["ws_auroc"]
    table = (runs / "table.txt").read_text()
    assert "within-slide AUROC" in table and "A-unw" in table
