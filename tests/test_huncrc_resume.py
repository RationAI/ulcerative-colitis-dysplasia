"""A fold that crashes in the middle of training resumes after its last epoch.

The crash is an exception raised at the start of epoch 3 inside the real
runner (``within_slide_cv.main``, in process), so the fold dies with two
epochs complete and no done.json. Restarted, it must resume after epoch 2 and
end with run files identical bit for bit to an uninterrupted run.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest
import torch

from tests.huncrc_synth import REPO, build


if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


class SimulatedCrashError(RuntimeError):
    pass


def _runner() -> ModuleType:
    path = REPO / "scripts" / "huncrc" / "within_slide_cv.py"
    spec = importlib.util.spec_from_file_location("within_slide_cv_resume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crash_mid_fold_resumes_after_the_last_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.test_huncrc_reference_equivalence import _diff_fold

    c = build(tmp_path / "synth")
    wcv = _runner()

    def train(runs: Path) -> None:
        wcv.main([str(a) for a in (
            "train", "--train-cache", c["train_within_slide"],
            "--prelim-all-cache", c["test_preliminary_all"], "--runs", runs,
            "--tiling", c["tiling"], "--slide-csv", c["slide_csv"],
            "--variants", "D", "--folds", 0, "--epochs", 3, "--neg-cap", 20,
            "--threads", 1,
        )])  # fmt: skip

    train(tmp_path / "full")

    class CrashingEpochFiles(wcv.EpochFiles):
        def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
            if trainer.current_epoch == 2:
                raise SimulatedCrashError("killed at the start of epoch 3")

    with monkeypatch.context() as m:
        m.setattr(wcv, "EpochFiles", CrashingEpochFiles)
        with pytest.raises(SimulatedCrashError):
            train(tmp_path / "crashed")
    d = tmp_path / "crashed" / "D" / "fold0"
    assert not (d / "done.json").exists()
    assert not (d / "epoch_02.json").exists()
    assert torch.load(d / "checkpoints" / "last.ckpt", weights_only=False)["epoch"] == 1
    capsys.readouterr()
    train(tmp_path / "crashed")
    assert "D/fold0: start (resume after epoch 2)" in capsys.readouterr().out
    assert _diff_fold(tmp_path / "full" / "D" / "fold0", d) == []
