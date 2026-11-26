"""Assembly of MLflow params/metrics from saved runs (log_to_mlflow.py), no server."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tests.huncrc_synth import REPO


def _load() -> ModuleType:
    path = REPO / "scripts" / "huncrc" / "log_to_mlflow.py"
    spec = importlib.util.spec_from_file_location("log_to_mlflow", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["log_to_mlflow"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load()
PRELIM_KEYS = [
    "ws_auroc",
    "spec_normal_pos",
    "slide_auroc_mean_all",
    "frac_above_carcinoma",
]
SHA = "fe0d8633f8df2e384baab43b94e72436ba9c5c3e"
HOME = "/home/jovyan/results"  # what the report records; the files are in tmp


def _w(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def make_set(
    root: Path,
    sub: str,
    head: str | None,
    counts: bool = True,
    smoke: int | None = None,
) -> None:
    """5 finished folds with 3 epochs; also the files that must never be uploaded."""
    for k in range(5):
        d = root / sub / f"fold{k}"
        done = {"variant": sub, "fold": k, "best_epoch": 2, "threshold": 0.4 + k / 100,
                "class_weights": None, "epochs": 3,
                "config": {"lr": 1e-4, "batch_size": 512, "epochs": 3, "dropout": 0.5,
                           "weight_decay": 0.01, "seed": k, "super_batch": 65536}}  # fmt: skip
        if head:
            done["head"] = head
        _w(d / "done.json", done)
        for e in range(3):
            _w(d / f"epoch_{e:02d}.json", {"epoch": e + 1, "train_loss": 1 / (e + 1),
               "val_auroc": 0.9 + e / 100, "val_auprc": 0.8 + e / 100, "seconds": 10.0 * (e + 1)})  # fmt: skip
            (d / f"epoch_{e:02d}.pt").write_bytes(b"x")
            np.save(d / f"epoch_{e:02d}_val.npy", np.zeros(3))
        np.save(d / "prelim_all_scores.npy", np.zeros(3))
        if counts:  # Step 8/9 layout has counts.json, Step 4a does not
            _w(
                d / "counts.json",
                {"neg_cap": 5000, "cap_seed": 0, "smoke_tiles_per_slide": smoke},
            )


def report_entry(offset: float) -> dict[str, Any]:
    val = {
        f"fold{k}": {
            "best_epoch": 2, "threshold": 0.4 + k / 100,
            "new": {"auroc": 0.90 + offset + k / 100, "auprc": 0.8, "sensitivity": 0.7,
                    "specificity": 0.6, "f1": 0.5, "slide_auroc": float("nan"),
                    "prevalence": 0.5, "n_tiles": 100},
            "new_ci": {"auroc": [0.85, 0.95]},
            "old": {"auroc": 0.99, "auprc": 0.98, "sensitivity": 0.7, "specificity": 0.9,
                    "f1": 0.8, "slide_auroc": 1.0 if k else float("nan"),
                    "prevalence": 0.6, "n_tiles": 80},
            "old_ci": {},
            "null_auroc": [0.5, 0.01], "null_auprc": [0.45, 0.02],
        }
        for k in range(5)
    }  # fmt: skip
    per_fold = [dict.fromkeys(PRELIM_KEYS, 0.5 + offset + k / 10) for k in range(5)]
    summary = {key: {"mean": 0.7, "sd": 0.15, "ci": [0.6, 0.8]} for key in PRELIM_KEYS}
    return {
        "folds": [0, 1, 2, 3, 4], "val": val,
        "prelim": {"per_fold": per_fold, "summary": summary,
                   "per_slide_ws_auroc": [{"016": 0.9 + k / 100, "056": 0.8} for k in range(5)],
                   "n": {"positive_slides": 15, "crc_slides_with_carcinoma": ["016", "096"]}},
    }  # fmt: skip


@pytest.fixture
def cohort(tmp_path: Path) -> dict[str, Any]:
    """Step 4a and 8 runs 'copied' to tmp/share; the report names their home paths."""
    share = tmp_path / "share"
    s8 = share / "step8"
    make_set(s8 / "runs", "B", head=None)  # Step-8 runs predate the "head" field
    _w(s8 / "runs" / "run_config.json", {"smoke_tiles_per_slide": "None"})
    (s8 / "run.log").write_text("[x] run_cv.sh train: threads=1 git=0695467\n")
    s9 = share / "step9_mlp"
    make_set(s9 / "runs", "E", head="mlp")
    (s9 / "run.log").write_text(f"[job] commit {SHA} msg\n[x] git=fe0d863\n")
    s4 = share / "step4a"
    make_set(s4, "a", head=None, counts=False)
    (s4 / "train.log").write_text("[15:44:31] a/fold0: train 10 tiles\n")
    rep = s9 / "report.json"
    _w(rep, {"model_sets": {"A-unw": report_entry(0.0), "B": report_entry(0.01),
                            "E": report_entry(0.02)}, "counts": {},
             "sources": {"A-unw": f"{HOME}/step4a/a", "B": f"{HOME}/step8/runs/B",
                         "E": str(s9 / "runs" / "E")}})  # fmt: skip
    (s9 / "table.txt").write_text("table\n")
    return {"report": rep, "map": {HOME: str(share)}, "share": share}


def plans(c: dict[str, Any], sets: list[str], **kw: Any) -> list[Any]:
    return L.build_plans(c["report"], sets, c["report"].with_name("table.txt"),
                         kw.get("git", {}), c["map"])  # fmt: skip


def test_params_and_metrics_from_saved_runs(cohort: dict[str, Any]) -> None:
    (b,) = plans(cohort, ["B"])
    assert b.name == "B"
    p = b.params
    assert p["head"] == "linear" and p["class_weighting"] == "none"
    assert p["within_slide_negatives"] == "true" and p["exclude_carcinoma"] == "true"
    assert p["excluded_slides"] == "043 135 140" and p["min_coverage"] == "0"
    assert p["label_rules_fingerprint"] == "729e9556"  # same as the real Step-8 cache
    assert p["neg_cap_per_slide"] == "5000" and p["epochs"] == "3"
    assert p["lr"] == "0.0001" and p["batch_size"] == "512"
    assert p["dropout"] == "0.5" and p["optimizer"] == "AdamW"
    assert p["seed"] == "fold index (0-4)" and p["class_weights_per_fold"] == "none"
    assert p["embedding_model"] == "prov-gigapath" and p["embedding_dim"] == "1536"
    # recorded at run start, but a notebook run never proves a clean tree
    assert p["git_commit"] == "0695467" and p["git_dirty"] == "unknown"
    assert p["git_commit_source"] == "run-log-head" == b.tags["git_commit_source"]
    assert "not established" in p["git_dirty_basis"]
    assert b.tags["mlflow.source.git.commit"] == "0695467"
    assert p["prelim_n_crc_slides_with_carcinoma"] == "016 096"
    assert p["runs_dir"] == f"{HOME}/step8/runs/B"  # as the report recorded it
    assert p["runs_dir_copy"] == str(cohort["share"] / "step8" / "runs" / "B")
    assert all(len(v) <= L.MAX_PARAM_LEN for v in p.values())
    t = b.tags
    assert t["step"] == "8" and t["origin"] == "jupyter" and t["smoke"] == "false"
    assert t["branch"] == "feature/huncrc-tile-classifier"

    aurocs = [0.91 + k / 100 for k in range(5)]
    m, s = b.metrics, b.series
    # headline = mean over folds; X_fold = per fold with step = fold; X_sd
    assert m["val_new_auroc"] == pytest.approx(np.mean(aurocs))
    assert m["val_new_auroc_sd"] == pytest.approx(np.std(aurocs, ddof=1))
    assert [st for st, _ in s["val_new_auroc_fold"]] == [0, 1, 2, 3, 4]
    assert [v for _, v in s["val_new_auroc_fold"]] == pytest.approx(aurocs)
    assert m["val_new_auroc_ci_low"] == 0.85
    assert m["val_threshold"] == pytest.approx(0.42) and m["val_best_epoch"] == 2
    assert m["val_null_auroc"] == 0.5 and m["val_null_auprc"] == 0.45
    # test_preliminary: the report's summary mean/sd/CI, per fold from per_fold
    assert m["prelim_ws_auroc"] == 0.7 and m["prelim_ws_auroc_sd"] == 0.15
    assert m["prelim_ws_auroc_ci_low"] == 0.6 and m["prelim_ws_auroc_ci_high"] == 0.8
    assert [v for _, v in s["prelim_ws_auroc_fold"]] == pytest.approx(
        [0.51 + k / 10 for k in range(5)]
    )
    # NaN never logged: all-NaN metric dropped, one NaN fold left out of the series
    assert "val_new_slide_auroc" not in m and "val_new_slide_auroc_fold" not in s
    assert [st for st, _ in s["val_old_slide_auroc_fold"]] == [1, 2, 3, 4]
    assert m["val_old_slide_auroc"] == 1.0
    assert "val_old_slide_auroc_fold@0" in b.skipped_metrics
    # per-epoch curves, step = epoch
    assert [st for st, _ in s["fold3_epoch_val_auroc"]] == [1, 2, 3]
    assert len([k for k in s if k.startswith("fold")]) == 5 * 4

    assert set(b.artifacts) == {"table.txt", "report.json", "per_slide_ws_auroc.csv",
                                "val_per_fold.csv", "prelim_per_fold.csv",
                                "epochs.csv", "folds.json"}  # fmt: skip
    assert b.artifacts["per_slide_ws_auroc.csv"].splitlines()[0] == (
        "slide,fold0,fold1,fold2,fold3,fold4,mean"
    )
    assert len(b.artifacts["epochs.csv"].splitlines()) == 1 + 5 * 3
    assert all(Path(a).suffix in {".txt", ".json", ".csv"} for a in b.artifacts)


def test_cluster_job_mlp_and_step4a_provenance(cohort: dict[str, Any]) -> None:
    a, e = plans(cohort, ["A-unw", "E"])
    assert e.params["head"] == "mlp" and e.params["class_weighting"] == "balanced"
    assert e.tags["origin"] == "cluster-job" and e.params["git_commit"] == SHA
    assert e.tags["step"] == "9" and e.tags["mlflow.source.git.commit"] == SHA
    assert (
        e.params["git_dirty"] == "false"
        and e.params["git_commit_source"] == "job-clone"
    )
    assert "runs_dir_copy" not in e.params  # read where the report says
    assert a.params["within_slide_negatives"] == "false"
    assert a.params["label_rules_fingerprint"] == "2b3fa61d"  # default rules
    assert a.params["git_commit"] == "unrecorded" and "git_note" in a.params
    assert a.params["git_dirty"] == "unknown"
    assert a.params["git_commit_source"] == "unrecorded"
    assert "mlflow.source.git.commit" not in a.tags
    assert a.params["neg_cap_per_slide"] == "none" and a.tags["step"] == "4a"
    (a2,) = plans(cohort, ["A-unw"], git={"A-unw": "a9e6511"})
    assert a2.params["git_commit"] == "a9e6511" and a2.params["git_dirty"] == "unknown"
    assert a2.tags["git_commit_source"] == "retrospective"
    assert a2.params["git_commit_recorded"] == "unrecorded"
    assert "mlflow.source.git.commit" not in a2.tags  # not claimed as the run's source


def test_source_map() -> None:
    m = {"/h/r": "/s", "/h/r/2026": "/s/x"}
    assert L.map_source("/h/r/2026/a", m) == "/s/x/a"  # longest prefix wins
    assert L.map_source("/h/r/B", m) == "/s/B"
    assert L.map_source("/h/rx/B", m) == "/h/rx/B"  # whole path components only
    assert L.map_source("/mnt/p/D", m) == "/mnt/p/D"


def test_refusals(cohort: dict[str, Any]) -> None:
    share = cohort["share"]
    d = share / "step9_mlp" / "runs" / "E" / "fold3"
    _w(d / "counts.json", {"neg_cap": 5000, "smoke_tiles_per_slide": 300})
    with pytest.raises(L.RefuseError, match="smoke"):
        plans(cohort, ["E"])
    _w(d / "counts.json", {"neg_cap": 5000, "smoke_tiles_per_slide": None})
    (d / "epoch_02.json").unlink()
    with pytest.raises(L.RefuseError, match="epoch files"):
        plans(cohort, ["E"])
    (share / "step8" / "runs" / "B" / "fold4" / "done.json").unlink()
    with pytest.raises(L.RefuseError, match="unfinished"):
        plans(cohort, ["B"])
    rep = json.loads(cohort["report"].read_text())
    rep["model_sets"]["A-unw"]["val"]["fold1"]["threshold"] = 0.99
    _w(cohort["report"], rep)
    with pytest.raises(L.RefuseError, match="disagree"):
        plans(cohort, ["A-unw"])
    rep["model_sets"]["A-unw"]["folds"] = [0]  # a partial (e.g. smoke) report
    _w(cohort["report"], rep)
    with pytest.raises(L.RefuseError, match="need all 5"):
        plans(cohort, ["A-unw"])
    with pytest.raises(L.RefuseError, match="not in"):
        plans(cohort, ["C"])
    with pytest.raises(L.RefuseError, match="does not exist"):  # no source map
        L.build_plans(cohort["report"], ["B"], None, {}, {})


def test_refuses_smoke_path(tmp_path: Path) -> None:
    root = tmp_path / "step9_mlp_smoke" / "runs"
    make_set(root, "D", head="mlp")
    rep = tmp_path / "report.json"
    _w(rep, {"model_sets": {"D": report_entry(0.0)}, "sources": {"D": str(root / "D")}})
    with pytest.raises(L.RefuseError, match="smoke"):
        L.build_plans(rep, ["D"], None, {})


def test_dry_run_never_imports_mlflow(
    cohort: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # any import of mlflow would fail
    for name in [m for m in sys.modules if m.startswith(("rationai", "ml.huncrc"))]:
        monkeypatch.delitem(sys.modules, name)
    fresh = _load()  # import the script itself with mlflow blocked, too
    fresh.main(["--dry-run", "--report", str(cohort["report"]), "--sets", "B", "E",
            "--source-map", f"{HOME}={cohort['share']}"])  # fmt: skip
    out = capsys.readouterr().out
    assert "2 runs, no nested runs: B, E" in out
    assert "RUN 'E'" in out and "nested under" not in out
    assert "val_new_auroc_fold: 0:0.92" in out
    assert ".pt " not in out and ".npy" not in out


@pytest.fixture
def local_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A throw-away SQLite tracking store (never the server)."""
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
                "MLFLOW_TRACKING_TOKEN"):  # fmt: skip
        monkeypatch.delenv(var, raising=False)
    client = mlflow.tracking.MlflowClient(uri)
    exp = client.create_experiment(L.EXPERIMENT_NAME, (tmp_path / "art").as_uri())
    return client, exp


def test_log_plans_to_local_store(cohort: dict[str, Any], local_store: Any) -> None:
    client, exp = local_store
    b, e = plans(cohort, ["B", "E"])
    L.log_plans([b, e], exp, allow_duplicate=False)

    runs = client.search_runs([exp])
    assert sorted(r.info.run_name for r in runs) == ["B", "E"]
    assert all("mlflow.parentRunId" not in r.data.tags for r in runs)
    rb = next(r for r in runs if r.info.run_name == "B")
    assert rb.data.params["head"] == "linear" and rb.data.tags["step"] == "8"
    assert rb.info.status == "FINISHED"
    assert rb.data.metrics["prelim_ws_auroc"] == 0.7  # the headline, not a fold
    assert rb.data.metrics["val_new_auroc"] == pytest.approx(0.93)
    hist = client.get_metric_history(rb.info.run_id, "val_new_auroc_fold")
    assert [h.step for h in sorted(hist, key=lambda h: h.step)] == [0, 1, 2, 3, 4]
    curve = client.get_metric_history(rb.info.run_id, "fold0_epoch_val_auroc")
    assert sorted(h.step for h in curve) == [1, 2, 3]
    assert {a.path for a in client.list_artifacts(rb.info.run_id)} == set(b.artifacts)

    with pytest.raises(SystemExit, match="already logged"):  # same key -> refused
        L.log_plans([b], exp, allow_duplicate=False)
    other = client.create_experiment("something else")
    with pytest.raises(SystemExit, match="expected"):  # wrong experiment -> refused
        L.log_plans([b], other, allow_duplicate=False)


def test_check_mlflow_is_read_only_and_hides_credentials(
    local_store: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, exp = local_store
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "s3cret-value")
    assert L.check_mlflow(exp) == 0
    out = capsys.readouterr().out
    assert f"name={L.EXPERIMENT_NAME!r}" in out and "artifact_location = file:" in out
    assert "MLFLOW_TRACKING_PASSWORD: set" in out and "s3cret" not in out
    assert "MLFLOW_TRACKING_USERNAME: not set" in out
    assert client.search_runs([exp]) == []  # nothing written
    assert L.check_mlflow(client.create_experiment("other")) == 3
    assert L.check_mlflow("999999") == 2
    assert "FAILED" in capsys.readouterr().out


def test_login_prompts_and_sets_env_of_this_process_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for var in L.AUTH_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "old-token")  # --login wins
    prompts: list[str] = []

    def answer(value: str) -> Any:
        def ask(q: str) -> str:
            prompts.append(q)
            return value

        return ask

    mode = L.setup_tracking(
        "https://example.invalid", login=True,
        ask=answer(" me@example.org "), ask_hidden=answer("typed-s3cret"),
    )  # fmt: skip
    assert "--login" in mode and "s3cret" not in mode
    assert any("hidden" in q for q in prompts)
    import os

    assert os.environ["MLFLOW_TRACKING_URI"] == "https://example.invalid"
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "me@example.org"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "typed-s3cret"
    assert "MLFLOW_TRACKING_TOKEN" not in os.environ
    assert "s3cret" not in capsys.readouterr().out
    assert L._mask("401 for typed-s3cret") == "401 for ***"
    with pytest.raises(SystemExit, match="required"):
        L.setup_tracking(
            "https://example.invalid", True, ask=lambda q: "x", ask_hidden=lambda q: ""
        )


def test_token_from_env_and_missing_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in L.AUTH_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "adryana")  # alone: not enough
    with pytest.raises(SystemExit, match="--login"):
        L.setup_tracking(L.PUBLIC_MLFLOW, login=False)
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "tok")
    assert "MLFLOW_TRACKING_TOKEN" in L.setup_tracking(L.PUBLIC_MLFLOW, login=False)
    import os

    assert "MLFLOW_TRACKING_USERNAME" not in os.environ
    assert os.environ["MLFLOW_TRACKING_URI"] == L.PUBLIC_MLFLOW


def test_dry_run_refuses_login(cohort: dict[str, Any]) -> None:
    with pytest.raises(SystemExit):
        L.main(["--dry-run", "--login", "--report", str(cohort["report"])])


def test_check_mlflow_cli_with_token(
    local_store: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, exp = local_store
    import os

    uri = os.environ["MLFLOW_TRACKING_URI"]
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "env-t0ken")
    with pytest.raises(SystemExit) as ex:
        L.main(["--check-mlflow", "--tracking-uri", uri, "--experiment-id", exp])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "credentials: none needed (local store)" in out
    assert "t0ken" not in out and "artifact_location = file:" in out


def test_dirty_flag_from_run_log(tmp_path: Path) -> None:
    (tmp_path / "runs" / "B").mkdir(parents=True)
    (tmp_path / "run.log").write_text("[x] run_cv.sh train: git=0695467+dirty\n")
    prov = L.code_provenance(tmp_path / "runs" / "B")
    assert prov["git_dirty"] == "true" and prov["git_commit_source"] == "run-log-head"
