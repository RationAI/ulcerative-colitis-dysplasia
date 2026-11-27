"""Hydra entry point ml.huncrc_cv and the optional MLflow logger (no training)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from ml import huncrc_cv as H
from ml import huncrc_tracking as T
from tests.huncrc_synth import REPO


CONFIGS = str(REPO / "configs")


def cfg(*overrides: str, hydra_cfg: bool = False) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        return compose(
            "huncrc_tile",
            overrides=["data.out=/mnt/projects/x/run", *overrides],
            return_hydra_config=hydra_cfg,
        )


@pytest.fixture(autouse=True)
def _isolated_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """No credentials; env and mlflow's global tracking URI restored afterwards."""
    import mlflow.tracking._tracking_service.utils as tracking_utils

    for var in T.AUTH_VARS:
        monkeypatch.delenv(var, raising=False)
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_HTTP_REQUEST_TIMEOUT",
                "MLFLOW_HTTP_REQUEST_MAX_RETRIES"):  # fmt: skip
        if var in os.environ:
            monkeypatch.setenv(var, os.environ[var])
        else:
            monkeypatch.delenv(var, raising=False)
    # lightning's MLFlowLogger calls mlflow.set_tracking_uri (process-global)
    monkeypatch.setattr(tracking_utils, "_tracking_uri", tracking_utils._tracking_uri)


@pytest.mark.parametrize(
    ("variant", "head", "weighting"),
    [("B", "linear", "none"), ("C", "linear", "balanced"),
     ("D", "mlp", "none"), ("E", "mlp", "balanced")],
)  # fmt: skip
def test_variants_compose_and_check(variant: str, head: str, weighting: str) -> None:
    c = cfg(f"experiment=huncrc/{variant}")
    assert c.model_set == variant and c.metadata.run_name == variant
    assert c.head.kind == head and c.training.class_weighting == weighting
    assert c.metadata.experiment_name == "Ulcerative Colitis Dysplasia"
    assert c.logger._target_ == "ml.huncrc_tracking.OptionalMLFlowLogger"
    assert c.logger.experiment_name == c.metadata.experiment_name  # house mapping
    assert c.logger.log_model is False
    assert H.check_config(c) == variant
    OmegaConf.to_container(c, resolve=True)  # everything resolves


def test_required_fields_fail_fast() -> None:
    with pytest.raises(Exception, match="experiment"):
        OmegaConf.to_container(cfg(), resolve=True)  # no experiment chosen
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        c = compose("huncrc_tile", overrides=["experiment=huncrc/B"])
    assert OmegaConf.is_missing(c.data, "out")


def test_check_config_refusals() -> None:
    with pytest.raises(SystemExit, match="is model set D"):
        H.check_config(cfg("experiment=huncrc/B", "huncrc/head@head=mlp"))
    with pytest.raises(SystemExit, match="does not take"):
        H.check_config(cfg("experiment=huncrc/B", "training.lr=0.001"))
    with pytest.raises(SystemExit, match="smoke"):
        H.check_config(cfg("experiment=huncrc/D", "training.smoke_tiles_per_slide=300"))
    c = cfg("experiment=huncrc/D", "training.smoke_tiles_per_slide=300",
            "metadata.experiment_name=HunCRC smoke")  # fmt: skip
    assert H.check_config(c) == "D"  # another experiment is fine
    c = cfg("experiment=huncrc/D", "training.smoke_tiles_per_slide=300",
            "mlflow.enabled=false")  # fmt: skip
    assert H.check_config(c) == "D"


def test_label_rules_warn_not_refuse() -> None:
    c = cfg("experiment=huncrc/C")
    assert H.label_rules_warning(c) is None
    train, report = H.build_calls(c, "C", "/c/t", "/c/p")
    assert "--label-rules-from-cache" not in train + (report or [])  # = run_cv.sh
    c = cfg("experiment=huncrc/C", "labels.min_coverage=0.5",
            "labels.exclude_carcinoma=false")  # fmt: skip
    assert H.check_config(c) == "C"  # allowed
    w = H.label_rules_warning(c)
    assert w and "WARNING" in w and "min_coverage: 0.5   (Step 8: 0.0)" in w
    assert "exclude_carcinoma: False   (Step 8: True)" in w
    assert "within_slide_negatives" not in w  # only what differs
    train, report = H.build_calls(c, "C", "/c/t", "/c/p")
    assert train[-1] == "--label-rules-from-cache" and report
    assert report[-1] == "--label-rules-from-cache"
    with pytest.raises(ValueError, match="min_coverage"):
        H.check_config(cfg("experiment=huncrc/C", "labels.min_coverage=2"))


def test_test_final_is_refused() -> None:
    with pytest.raises(SystemExit, match="test_final"):
        H.check_config(cfg("experiment=huncrc/B", "data.cache_root=/x/test_final"))
    with pytest.raises(SystemExit, match="test_final"):
        H.check_config(cfg("experiment=huncrc/B"), ["+splits=[train,test_final]"])


def test_within_slide_cv_trains_on_the_cache_rules(tmp_path: Path) -> None:
    import pandas as pd

    from ml.data.huncrc_features import build_features
    from ml.data.huncrc_labels import load_slide_table, rules_fingerprint
    from tests.huncrc_synth import build, run_script

    c = build(tmp_path)
    table = load_slide_table(pd.read_parquet(c["tiling"] / "slides" / "slides.parquet"),
                             pd.read_csv(c["slide_csv"], dtype=str))  # fmt: skip
    rules = H.label_config(cfg("experiment=huncrc/B", "labels.min_coverage=0.5"))
    other = build_features(tmp_path / "emb", table, ["train"], rules, tmp_path / "cache",
                           dim=16, annotations=c["annotations"])  # fmt: skip
    args = ["train", "--train-cache", other, "--prelim-all-cache",
            c["test_preliminary_all"], "--runs", tmp_path / "runs", "--tiling",
            c["tiling"], "--slide-csv", c["slide_csv"], "--variants", "B",
            "--folds", "0", "--epochs", "1"]  # fmt: skip
    proc = subprocess.run([sys.executable, str(REPO / "scripts/huncrc/within_slide_cv.py"),
                           *map(str, args)], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO)}, check=False)  # fmt: skip
    assert proc.returncode != 0 and "StaleCacheError" in proc.stderr  # Step 8 only
    proc = run_script("within_slide_cv.py", *args, "--label-rules-from-cache")
    assert (
        "WARNING: label rules differ" in proc.stdout
        and "min_coverage=0.5" in proc.stdout
    )
    rc = json.loads((tmp_path / "runs" / "run_config.json").read_text())
    assert rc["label_rules_fingerprint"] == rules_fingerprint(rules) != "729e9556"


def test_no_credential_can_reach_the_uploaded_config() -> None:
    c = cfg("experiment=huncrc/B")
    with open_dict(c):
        c.logger.tags["x"] = "${oc.env:MLFLOW_TRACKING_PASSWORD}"
    with pytest.raises(SystemExit, match="environment interpolation"):
        H.check_config(c)
    c = cfg("experiment=huncrc/B", "+mlflow.token=abc")
    with pytest.raises(SystemExit, match="credential"):
        H.check_config(c)
    with pytest.raises(SystemExit, match="credential"):
        H.check_config(cfg("experiment=huncrc/B"), ["+api_key=abc"])
    # the shipped tree itself is clean
    assert T.find_secrets(H._items(cfg("experiment=huncrc/E"))) == []


def test_calls_match_run_cv_sh() -> None:
    c = cfg("experiment=huncrc/D", "training.smoke_tiles_per_slide=300",
            "evaluation.folds=[0]", "training.epochs=1", "mlflow.enabled=false",
            "evaluation.other_runs=[/mnt/projects/a/runs]")  # fmt: skip
    train, report = H.build_calls(c, "D", "/c/train", "/c/prelim")
    common = ["--train-cache", "/c/train", "--prelim-all-cache", "/c/prelim",
              "--runs", "/mnt/projects/x/run/runs",
              "--tiling", "/mnt/projects/huncrc_dysplasia/preprocessing/tiling_200",
              "--slide-csv", "/mnt/data/Public/Public/colorectum/TCIA_HunCRC/"
              "slide_level_annotations.csv"]  # fmt: skip
    assert train == ["train", *common, "--variants", "D", "--folds", "0",
                     "--epochs", "1", "--neg-cap", "5000", "--cap-seed", "0",
                     "--smoke-tiles-per-slide", "300"]  # fmt: skip
    assert report == ["report", *common,
                      "--annotations", "/mnt/projects/huncrc_dysplasia/preprocessing/annotations",
                      "--n-boot", "1000", "--out", "/mnt/projects/x/run",
                      "--other-runs", "/mnt/projects/a/runs"]  # fmt: skip
    # within_slide_cv's own parser accepts exactly these
    wcv = H._script("within_slide_cv")
    import argparse

    seen: list[argparse.Namespace] = []
    orig = argparse.ArgumentParser.parse_args

    def spy(self: argparse.ArgumentParser, args: Any = None, ns: Any = None) -> Any:
        seen.append(orig(self, args, ns))
        raise SystemExit(0)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(argparse.ArgumentParser, "parse_args", spy)
        for argv in (train, report):
            with pytest.raises(SystemExit):
                wcv.main(argv)
    assert seen[0].variants == ["D"] and seen[0].smoke_tiles_per_slide == 300
    assert seen[1].other_runs == [Path("/mnt/projects/a/runs")]


# ------------------------------------------------------------------ logger


def _logger(**kw: Any) -> T.OptionalMLFlowLogger:
    return T.OptionalMLFlowLogger(experiment_name="t", run_name="r", **kw)


def test_logger_off_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("network call")

    import mlflow.utils.rest_utils as rest
    import requests

    monkeypatch.setattr(rest, "http_request", boom)  # every REST call goes here
    monkeypatch.setattr(requests.Session, "request", boom)  # and any raw HTTP
    for lg in (
        _logger(enabled=False),  # cluster job
        _logger(tracking_uri="https://mlflow.invalid"),
    ):  # no credentials
        assert not lg.active
        lg.log_metrics({"a": 1.0}, step=0)  # no-ops, no exception
        lg.log_series("s", [(0, 1.0)])
        lg.log_stream("text")
        lg.log_hyperparams({"a": 1})
    assert "username alone" in _logger(tracking_uri="https://mlflow.invalid").reason
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "adryana")
    assert not _logger(tracking_uri="https://mlflow.invalid").active


def test_logger_unreachable_server_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "t0k")
    lg = _logger(tracking_uri="http://127.0.0.1:9", timeout_s=2)  # nothing listens
    assert not lg.active and "server not usable" in lg.reason
    assert "t0k" not in lg.reason


def test_logger_local_store_and_failure_midway(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import mlflow

    uri = f"sqlite:///{tmp_path / 'm.db'}"
    lg = _logger(tracking_uri=uri, log_model="all")
    assert lg.active and lg._log_model is False  # checkpoints never uploaded
    lg.log_series("fold0_epoch_val_auroc", [(1, 0.9), (2, 0.91)])
    lg.log_series(
        "fold0_epoch_val_auroc", [(2, 0.91), (3, 0.92)]
    )  # step 2 not repeated
    client = mlflow.MlflowClient(uri)
    assert lg.run_id is not None
    hist = client.get_metric_history(lg.run_id, "fold0_epoch_val_auroc")
    assert sorted(h.step for h in hist) == [1, 2, 3]

    def fail(*a: Any, **k: Any) -> None:
        raise ConnectionError("server went away")

    lg.experiment.log_batch = fail
    lg.log_series("x", [(0, 1.0)])  # swallowed, logging turned off
    assert not lg.active and "server went away" in lg.reason
    lg.log_metrics({"a": 1.0})  # still no exception
    mlflow.end_run()


def test_autolog_upload_has_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What @autolog uploads, with a token in the environment, holds no secret."""
    import mlflow
    from rationai.mlkit import autolog

    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "sup3r-secret")
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'm.db'}"
    c = cfg("experiment=huncrc/E", f"mlflow.tracking_uri={uri}",
            "metadata.experiment_name=pytest", hydra_cfg=True)  # fmt: skip
    HydraConfig.instance().set_config(c)
    with open_dict(c):
        del c["hydra"]
    seen: dict[str, Any] = {}

    def body(config: DictConfig, logger: T.OptionalMLFlowLogger) -> None:
        seen["run"] = logger.run_id
        print("training would run here")

    autolog(body)(c)  # type: ignore[arg-type]
    mlflow.end_run()
    client = mlflow.MlflowClient(uri)
    run = client.get_run(seen["run"])
    assert run.info.run_name == "E" and run.data.params["config/head"] == "mlp"
    out = tmp_path / "dl"
    client.download_artifacts(seen["run"], "", str(out))
    texts = [p.read_text() for p in out.rglob("*") if p.is_file()]
    assert any("training would run here" in t for t in texts)  # console.log
    assert {p.name for p in (out / "configs").iterdir()} == {
        "hydra.yaml", "config.yaml", "config-resolved.yaml"}  # fmt: skip
    assert not any("sup3r-secret" in t for t in texts)
    assert not any(p.suffix in {".pt", ".ckpt", ".npy"} for p in out.rglob("*"))
    assert os.environ.get("MLFLOW_TRACKING_TOKEN") == "sup3r-secret"  # process env only


def test_within_slide_cv_refuses_mixed_embeddings(tmp_path: Path) -> None:
    """Train cache of one embedding + test_preliminary cache of another: refused."""
    import pandas as pd

    from ml.data.huncrc_features import build_features
    from ml.data.huncrc_labels import LabelConfig, load_slide_table
    from tests.huncrc_synth import build

    c = build(tmp_path)
    table = load_slide_table(pd.read_parquet(c["tiling"] / "slides" / "slides.parquet"),
                             pd.read_csv(c["slide_csv"], dtype=str))  # fmt: skip
    other = tmp_path / "embeddings" / "virchow2"
    other.mkdir(parents=True)
    for f in (tmp_path / "emb").glob("*.parquet"):
        (other / f.name).write_bytes(f.read_bytes())
    prelim = build_features(other, table, ["test_preliminary"], LabelConfig(),
                            tmp_path / "cache", dim=16, mode="all")  # fmt: skip
    args = ["train", "--train-cache", c["train_within_slide"], "--prelim-all-cache",
            prelim, "--runs", tmp_path / "runs", "--tiling", c["tiling"],
            "--slide-csv", c["slide_csv"], "--variants", "B", "--folds", "0",
            "--epochs", "1"]  # fmt: skip
    proc = subprocess.run([sys.executable, str(REPO / "scripts/huncrc/within_slide_cv.py"),
                           *map(str, args)], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO)}, check=False)  # fmt: skip
    assert proc.returncode != 0 and "EmbeddingMismatchError" in proc.stderr
    assert "holds virchow2 16-d" in proc.stderr and "expected emb 16-d" in proc.stderr
    assert not (tmp_path / "runs" / "B").exists()  # nothing trained
