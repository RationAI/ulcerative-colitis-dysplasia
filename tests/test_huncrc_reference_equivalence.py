"""The branch reproduces the reference implementation bit for bit.

The reference is commit 8b5955c (feature/huncrc-tile-classifier), checked out
read-only at ``$HUNCRC_REFERENCE`` (default: ``../ucd-reference-8b5955c`` next
to this repo); the test is skipped when it is missing and fails when it is at
another commit. Both implementations train model sets C (linear, balanced) and
D (MLP, unweighted) on fold 0 of the synthetic cohort for 3 epochs from the same
feature caches, then write the report. Every file the reference writes must be
identical in the branch run: training and validation rows, counts, scaler,
per-epoch metrics and validation scores, the saved weights, the best epoch,
threshold and config, the test_preliminary scores, report.json and table.txt.

The branch run is also done a second time with an interruption after epoch 1
(the fold's done.json is removed, then training is restarted for 3 epochs);
it must equal the uninterrupted run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tests.huncrc_synth import REPO, build


REF_COMMIT = "8b5955c"
REFERENCE = Path(
    os.environ.get("HUNCRC_REFERENCE", REPO.parent / f"ucd-reference-{REF_COMMIT}")
)
VARIANTS = ("C", "D")
EPOCHS, NEG_CAP, NBOOT = 3, 20, 50


def _reference_ok() -> bool:
    if not (REFERENCE / "scripts" / "huncrc" / "within_slide_cv.py").is_file():
        return False
    head = subprocess.run(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout  # fmt: skip
    assert head.startswith(REF_COMMIT), (
        f"{REFERENCE} is at {head[:12]}, not {REF_COMMIT}"
    )
    dirty = subprocess.run(["git", "-C", str(REFERENCE), "status", "--porcelain",
                            "--", "ml", "scripts"],
                           capture_output=True, text=True, check=True).stdout  # fmt: skip
    assert not dirty, f"{REFERENCE} has local changes:\n{dirty}"
    return True


pytestmark = pytest.mark.skipif(
    not _reference_ok(), reason=f"reference checkout {REFERENCE} not found"
)


def _wcv(root: Path, c: dict[str, Path], runs: Path, *args: object) -> str:
    """scripts/huncrc/within_slide_cv.py of ``root``, importing ``root``'s ml."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}
    env |= {"PYTHONPATH": str(root), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1"}  # fmt: skip
    cmd = [sys.executable, str(root / "scripts" / "huncrc" / "within_slide_cv.py"),
           *map(str, args), "--train-cache", c["train_within_slide"],
           "--prelim-all-cache", c["test_preliminary_all"], "--runs", runs,
           "--tiling", c["tiling"], "--slide-csv", c["slide_csv"], "--threads", "1"]  # fmt: skip
    proc = subprocess.run(list(map(str, cmd)), capture_output=True, text=True,
                          env=env, cwd=root, check=False)  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def _train(root: Path, c: dict[str, Path], runs: Path, epochs: int) -> None:
    _wcv(root, c, runs, "train", "--variants", *VARIANTS, "--folds", 0,
         "--epochs", epochs, "--neg-cap", NEG_CAP)  # fmt: skip


def _report(root: Path, c: dict[str, Path], runs: Path) -> None:
    _wcv(root, c, runs, "report", "--annotations", c["annotations"],
         "--n-boot", NBOOT)  # fmt: skip


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("refequiv")
    c = build(tmp / "synth")
    out = {"ref": tmp / "ref", "new": tmp / "new", "resumed": tmp / "resumed"}
    for name, root in (("ref", REFERENCE), ("new", REPO)):
        _train(root, c, out[name], EPOCHS)
        _report(root, c, out[name])
    # interrupted after epoch 1: the fold never finished, then training restarts
    _train(REPO, c, out["resumed"], 1)
    for v in VARIANTS:
        for name in ("done.json", "prelim_all_scores.npy"):
            (out["resumed"] / v / "fold0" / name).unlink()
    _train(REPO, c, out["resumed"], EPOCHS)
    _report(REPO, c, out["resumed"])
    return out


def _same_json(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _diff_fold(ref: Path, new: Path) -> list[str]:
    """Every file of the reference fold dir, compared bit for bit."""
    bad = []
    done = json.loads((ref / "done.json").read_text())
    for p in sorted(ref.iterdir()):
        if p.is_dir():  # Lightning's checkpoints/ of a branch run
            continue
        q = new / p.name
        if p.name.startswith("epoch_") and p.suffix == ".pt" and not q.exists():
            # only the best epoch's weights are needed downstream
            if p.name == f"epoch_{done['best_epoch'] - 1:02d}.pt":
                bad.append(f"{q} missing (best epoch weights)")
            continue
        if not q.exists():
            bad.append(f"{q} missing")
        elif p.suffix == ".npy":
            x, y = np.load(p), np.load(q)
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                bad.append(f"{p.name} differs")
        elif p.suffix == ".npz":
            x, y = np.load(p), np.load(q)
            if sorted(x) != sorted(y) or any(
                x[k].tobytes() != y[k].tobytes() for k in x
            ):
                bad.append(f"{p.name} differs")
        elif p.suffix == ".pt":
            x, y = (torch.load(f, weights_only=False) for f in (p, q))
            if (
                x["epoch"] != y["epoch"]
                or x["model"].keys() != y["model"].keys()
                or not all(
                    torch.equal(x["model"][k], y["model"][k]) for k in x["model"]
                )
            ):
                bad.append(f"{p.name} model differs")
            if "optimizer" in y and repr(x["optimizer"]["param_groups"]) != repr(
                y["optimizer"]["param_groups"]
            ):
                bad.append(f"{p.name} optimizer param groups differ")
            if "optimizer" in y and not all(
                torch.equal(
                    x["optimizer"]["state"][i][k], y["optimizer"]["state"][i][k]
                )
                for i in x["optimizer"]["state"]
                for k in x["optimizer"]["state"][i]
            ):
                bad.append(f"{p.name} optimizer state differs")
        elif p.suffix == ".json":
            x, y = (json.loads(f.read_text()) for f in (p, q))
            x.pop("seconds", None)  # wall time
            y.pop("seconds", None)
            if not _same_json(x, y):
                bad.append(f"{p.name}: {x} != {y}")
        else:
            bad.append(f"{p.name}: no comparison defined")
    return bad


@pytest.mark.parametrize("other", ["new", "resumed"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fold_files_identical(runs: dict[str, Path], variant: str, other: str) -> None:
    ref = runs["ref"] / variant / "fold0"
    assert _diff_fold(ref, runs[other] / variant / "fold0") == []


@pytest.mark.parametrize("other", ["new", "resumed"])
def test_report_identical(runs: dict[str, Path], other: str) -> None:
    def load(d: Path, name: str) -> str:
        return (d / name).read_text().replace(str(d), "<runs>")

    for name in ("report.json", "table.txt"):
        assert load(runs["ref"], name) == load(runs[other], name), name


def test_run_config_identical(runs: dict[str, Path]) -> None:
    keys = ("train_cache", "prelim_all_cache", "epochs", "neg_cap", "cap_seed",
            "smoke_tiles_per_slide", "label_rules_fingerprint", "embedding_model",
            "embedding_dim")  # fmt: skip
    x, y = (
        json.loads((runs[n] / "run_config.json").read_text()) for n in ("ref", "new")
    )
    assert {k: x[k] for k in keys} == {k: y[k] for k in keys}


def test_the_runs_did_train(runs: dict[str, Path]) -> None:
    """Guard against a vacuous pass: three epochs, distinct weights per epoch."""
    for v in VARIANTS:
        d = runs["ref"] / v / "fold0"
        assert sorted(p.name for p in d.glob("epoch_??.json")) == [
            f"epoch_{e:02d}.json" for e in range(EPOCHS)
        ]
        w = [torch.load(d / f"epoch_{e:02d}.pt", weights_only=False)["model"]
             for e in range(EPOCHS)]  # fmt: skip
        k = next(iter(w[0]))
        assert not torch.equal(w[0][k], w[-1][k])
