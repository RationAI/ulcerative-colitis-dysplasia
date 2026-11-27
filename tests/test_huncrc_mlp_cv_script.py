"""Smoke test: within_slide_cv.py with the MLP variants D/E, and a combined report."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.huncrc_synth import REPO, build, run_script


def test_mlp_variants_and_combined_report(tmp_path: Path) -> None:
    c = build(tmp_path)
    paths = ["--tiling", c["tiling"], "--slide-csv", c["slide_csv"]]
    caches = [
        "--train-cache", c["train_within_slide"],
        "--prelim-all-cache", c["test_preliminary_all"], *paths,
    ]  # fmt: skip
    bc, de = tmp_path / "bc_runs", tmp_path / "de_runs"
    run_script(
        "within_slide_cv.py", "train", *caches, "--runs", bc, "--variants", "B",
        "--folds", 0, 1, "--epochs", 1, "--neg-cap", 20,
    )  # fmt: skip
    proc = run_script(
        "within_slide_cv.py", "train", *caches, "--runs", de, "--variants", "D", "E",
        "--folds", 0, 1, "--epochs", 2, "--neg-cap", 20,
    )  # fmt: skip
    assert proc.stdout.count("DONE") == 4 and "head mlp" in proc.stdout
    for v, weighted in (("D", False), ("E", True)):
        done = json.loads((de / v / "fold0" / "done.json").read_text())
        assert done["head"] == "mlp" and (done["class_weights"] is not None) == weighted
        state = torch.load(de / v / "fold0" / "epoch_00.pt", weights_only=False)
        shapes = {k: tuple(t.shape) for k, t in state["model"].items()}
        assert shapes["hidden.0.weight"] == (512, 16)  # synthetic dim 16
        assert shapes["hidden.2.weight"] == (256, 512)
        assert shapes["proj.weight"] == (1, 256)
    assert (
        json.loads((bc / "B" / "fold0" / "done.json").read_text())["head"] == "linear"
    )

    # report: D/E from --runs, B from --other-runs, no Step-4a runs
    out = tmp_path / "report"
    run_script(
        "within_slide_cv.py", "report", *caches, "--runs", de, "--other-runs", bc,
        "--annotations", c["annotations"], "--n-boot", 10, "--out", out,
    )  # fmt: skip
    rep = json.loads((out / "report.json").read_text())
    assert set(rep["model_sets"]) == {"B", "D", "E"}
    assert rep["model_sets"]["D"]["head"] == "mlp"
    assert rep["model_sets"]["B"]["head"] == "linear"
    assert rep["sources"]["B"] == str(bc / "B") and rep["sources"]["E"] == str(de / "E")
    assert rep["model_sets"]["E"]["folds"] == [0, 1]
    table = (out / "table.txt").read_text()
    assert "D: " in table and "E: " in table and "B: " in table

    # the same variant in two runs dirs is refused instead of silently picking one
    proc2 = subprocess.run(
        [sys.executable, str(REPO / "scripts/huncrc/within_slide_cv.py"), "report",
         *map(str, caches), "--runs", str(de), "--other-runs", str(de),
         "--annotations", str(c["annotations"]), "--n-boot", "0"],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)}, check=False,
    )  # fmt: skip
    assert proc2.returncode != 0 and "several runs dirs" in proc2.stderr


@pytest.mark.parametrize("smoke", [True, False])
def test_submit_job_dry_run(smoke: bool) -> None:
    args = ["--dry-run", *(["--smoke"] if smoke else [])]
    out = run_script("submit_mlp_job.py", *args).stdout
    assert "public=True cpu=8 memory=32Gi shm=16Gi" in out  # public data by default
    assert (
        "git clone --depth 1 --single-branch -b feature/huncrc-tile-classifier "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir"
    ) in out
    assert out.index("run_cv.sh train") < out.index("run_cv.sh report")
    assert "uv sync --frozen" in out and "test_final" not in out
    res = "/mnt/projects/huncrc_dysplasia/tile_classifier/"
    if smoke:
        assert f"OUT={res}step9_mlp_smoke " in out and "FOLDS=0 EPOCHS=1" in out
        assert "SMOKE_TILES=300" in out and "huncrc-mlp-smoke" in out
    else:
        assert f"OUT={res}step9_mlp " in out and "FOLDS='0 1 2 3 4' EPOCHS=10" in out
        assert "SMOKE_TILES=''" in out and "VARIANTS='D E'" in out
    assert f"tee -a {res}step9_mlp{'_smoke' if smoke else ''}/run.log" in out


def test_submit_job_secure_opt_out() -> None:
    out = run_script("submit_mlp_job.py", "--dry-run", "--secure").stdout
    assert "public=False" in out


@pytest.mark.parametrize(
    "args",
    [
        ["--dry-run", "--cache-root", "/home/jovyan/huncrc_cache"],
        ["--dry-run", "--out", "/tmp/results"],
        ["--dry-run", "--other-runs", "/home/jovyan/step8/runs"],
        ["--dry-run", "--variants", "A-unw"],
    ],
)
def test_submit_job_refuses(args: list[str]) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/huncrc/submit_mlp_job.py"), *args],
        capture_output=True, text=True, cwd=REPO, check=False,
    )  # fmt: skip
    assert proc.returncode == 2, proc.stdout + proc.stderr
