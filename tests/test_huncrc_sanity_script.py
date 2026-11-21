"""Smoke test: scripts/huncrc/sanity_checks.py end to end on a synthetic cache."""

import json
from pathlib import Path

import pytest

from tests.huncrc_synth import build, run_script


def test_sanity_checks_script(tmp_path: Path) -> None:
    caches = build(tmp_path)
    out = tmp_path / "sanity.json"
    run_script(
        "sanity_checks.py",
        "--cache", caches["train"],
        "--out", out,
        "--overfit-epochs", 3,
        "--shuffled-epochs-linear", 1,
        "--shuffled-epochs-mlp", 1,
    )  # fmt: skip
    r = json.loads(out.read_text())
    # fold 0 = 3 slides validation, folds 1-4 = 12 slides training
    assert r["split"]["val_slides"] == 3 and r["split"]["train_slides"] == 12
    # majority class is dysplasia -> constant scores: AUROC 0.5, AUPRC = prevalence
    assert r["majority"]["auroc"] == 0.5
    assert r["majority"]["auprc"] == pytest.approx(r["split"]["val_prevalence"])
    assert r["all_dysplasia"]["recall"] == 1.0
    assert len(r["overfit"]["slides"]) == 5
    for kind in ("linear", "mlp"):
        assert 0.0 <= r["shuffled"][kind]["val_real"]["auroc"] <= 1.0


def test_sanity_checks_refuses_other_rules(tmp_path: Path) -> None:
    caches = build(tmp_path)
    meta = json.loads((caches["train"] / "meta.json").read_text())
    meta["rules_fingerprint"] = "00000000"
    (caches["train"] / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match="StaleCacheError"):
        run_script(
            "sanity_checks.py", "--cache", caches["train"], "--out", tmp_path / "x.json"
        )
