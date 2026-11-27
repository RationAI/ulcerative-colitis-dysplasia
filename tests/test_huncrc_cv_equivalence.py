"""The Hydra entry point and the existing CLI give the same run.

Model set C, fold 0, one epoch, a smoke subset, on the synthetic cohort with
1536-d embeddings. Path 1 is ``bash scripts/huncrc/run_cv.sh train`` + ``report``
(which calls build_feature_cache.py and within_slide_cv.py); path 2 is
``python -m ml.huncrc_cv experiment=huncrc/C ...`` with MLflow off. Each path
gets its own cache root and output dir, so both really build the caches and
train. Compared: cache keys and cache contents, the resolved training
configuration (run_config.json, counts.json, done.json incl. TrainConfig),
training rows, scaler, per-epoch metrics, test_preliminary scores and the
report.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tests.huncrc_synth import REPO, build


if TYPE_CHECKING:
    from pathlib import Path


VENV_PY = REPO / ".venv" / "bin" / "python"
pytestmark = pytest.mark.skipif(
    not VENV_PY.is_file() or shutil.which("bash") is None,
    reason="run_cv.sh needs bash and the repo .venv",
)
SMOKE, NBOOT = 25, 50


def _run(cmd: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO,
                          check=False)  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


def _digest(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.glob("*.npy"))}  # fmt: skip


def _json(p: Path, *, drop: tuple[str, ...] = (), roots: dict[str, str]) -> Any:
    text = p.read_text()
    for old, new in roots.items():
        text = text.replace(old, new)
    obj = json.loads(text)
    for k in drop:
        obj.pop(k, None)
    return obj


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("equiv")
    c = build(tmp / "synth", dim=1536)
    pre = tmp / "pre"  # the layout run_cv.sh expects under PRE
    (pre / "embeddings").mkdir(parents=True)
    (pre / "tiling_200").symlink_to(c["tiling"])
    (pre / "embeddings" / "prov-gigapath").symlink_to(tmp / "synth" / "emb")
    (pre / "annotations").symlink_to(c["annotations"])
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}
    env |= {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(REPO),
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:9"}  # fmt: skip

    cli = tmp / "cli"
    cli_env = env | {"PRE": str(pre), "SLIDE_CSV": str(c["slide_csv"]),
                     "CACHE_ROOT": str(cli / "cache"), "OUT": str(cli / "out"),
                     "VARIANTS": "C", "FOLDS": "0", "EPOCHS": "1",
                     "SMOKE_TILES": str(SMOKE), "N_BOOT": str(NBOOT),
                     "THREADS": "1", "A_RUNS": ""}  # fmt: skip
    for cmd in ("train", "report"):
        _run(["bash", "scripts/huncrc/run_cv.sh", cmd], cli_env)

    hyd = tmp / "hydra"
    log = _run([sys.executable, "-m", "ml.huncrc_cv", "experiment=huncrc/C",
                f"data.preprocessing={pre}", f"data.slide_csv={c['slide_csv']}",
                f"data.cache_root={hyd / 'cache'}", f"data.out={hyd / 'out'}",
                "evaluation.folds=[0]", "training.epochs=1",
                f"training.smoke_tiles_per_slide={SMOKE}",
                f"evaluation.n_boot={NBOOT}", "training.threads=1",
                "mlflow.enabled=false", f"hydra.run.dir={hyd}"], env)  # fmt: skip
    assert "logging OFF" in log and "WARNING" not in log
    return {"cli": cli, "hydra": hyd}


def test_same_cache_keys_and_contents(runs: dict[str, Path]) -> None:
    keys = {p: sorted(d.name for d in (runs[p] / "cache").iterdir()) for p in runs}
    assert keys["cli"] == keys["hydra"]
    assert len(keys["cli"]) == 2 and any("-train-r729e9556-" in k for k in keys["cli"])
    assert not any(
        "-e" in k.split("-r", 1)[1] for k in keys["cli"]
    )  # default embedding
    for key in keys["cli"]:
        a, b = (runs[p] / "cache" / key for p in ("cli", "hydra"))
        assert _digest(a) == _digest(b), key
        ma, mb = (json.loads((x / "meta.json").read_text()) for x in (a, b))
        assert ma == mb, key  # rules, counts, slides, sources: all of it


def test_same_training_configuration(runs: dict[str, Path]) -> None:
    roots = {str(runs["cli"]): "<root>", str(runs["hydra"]): "<root>"}
    ra, rb = (_json(runs[p] / "out" / "runs" / "run_config.json", roots=roots)
              for p in ("cli", "hydra"))  # fmt: skip
    assert ra == rb
    assert ra["variants"] == "['C']" and ra["epochs"] == "1" and ra["threads"] == "1"
    assert ra["label_rules_fingerprint"] == "729e9556"
    fa, fb = (runs[p] / "out" / "runs" / "C" / "fold0" for p in ("cli", "hydra"))
    for name in ("counts.json", "done.json"):
        assert _json(fa / name, roots=roots) == _json(fb / name, roots=roots), name
    done = _json(fa / "done.json", roots=roots)
    assert done["config"]["lr"] == 1e-4 and done["class_weights"] is not None


def test_same_metrics(runs: dict[str, Path]) -> None:
    fa, fb = (runs[p] / "out" / "runs" / "C" / "fold0" for p in ("cli", "hydra"))
    roots: dict[str, str] = {}
    ea, eb = (_json(f / "epoch_00.json", drop=("seconds",), roots=roots)
              for f in (fa, fb))  # fmt: skip
    assert ea == eb  # train_loss, val_auroc, val_auprc: bit-identical
    assert _digest(fa) == _digest(fb)  # rows, val scores, prelim scores
    za, zb = (np.load(f / "scaler.npz") for f in (fa, fb))
    assert np.array_equal(za["mean"], zb["mean"]) and np.array_equal(
        za["std"], zb["std"]
    )
    roots = {str(runs["cli"]): "<root>", str(runs["hydra"]): "<root>"}
    pa, pb = (_json(runs[p] / "out" / "report.json", roots=roots)
              for p in ("cli", "hydra"))  # fmt: skip
    assert pa == pb
    ta, tb = ((runs[p] / "out" / "table.txt").read_text().replace(str(runs[p]), "<root>")
              for p in ("cli", "hydra"))  # fmt: skip
    assert ta == tb
