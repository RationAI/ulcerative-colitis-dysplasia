r"""Step 10: log the final HunCRC model sets to MLflow, retrospectively.

Nothing is retrained or re-scored. Every number comes from the saved runs
(``done.json``, ``epoch_XX.json``, ``counts.json``) and from the combined
``report.json`` written by ``within_slide_cv.py report``.

    # 1. read-only: experiment name / artifact location / does auth work
    python scripts/huncrc/log_to_mlflow.py --check-mlflow --login
    # 2. print everything that would be created, touch nothing
    python scripts/huncrc/log_to_mlflow.py --dry-run --report <combined report.json> \
        --source-map /home/jovyan/huncrc_tile_classifier_results/2026-11-21=<share> ...

ONE MLflow run per model set (A-unw, A-bal, B, C, D, E), named by the set, no
nested runs. For every metric ``X``:
    X          the mean over the 5 folds (the headline value, step 0)
    X_fold     the per-fold values, step = fold index (0-4)
    X_sd       the sd over folds (ddof=1)
test_preliminary metrics also get ``X_ci_low`` / ``X_ci_high``, the slide
bootstrap CI of the mean from the report. The per-epoch curves are
``fold<k>_epoch_<name>`` with step = epoch. Artifacts are only .txt / .json /
.csv summaries: no caches, checkpoints or per-tile score arrays.

The report's ``sources`` say where each set's runs are. ``--source-map OLD=NEW``
rewrites a path prefix, for runs copied to the share.

Authentication (the server sits behind an OIDC login plugin). Run it from a
Jupyter terminal, because the prompt needs a TTY:
- ``--login`` prompts for the login email and the MLflow API access token
  (hidden, via getpass; not the SSO password), as the bone-marrow-fibrosis
  helper does. They are put in MLFLOW_TRACKING_USERNAME / _PASSWORD of this
  process only: never echoed, logged, stored, or passed as an argument.
- otherwise MLFLOW_TRACKING_TOKEN from the environment, if set;
- otherwise MLFLOW_TRACKING_USERNAME + _PASSWORD from the environment.
Only which of these was used is printed. The tracking URI defaults to the
public endpoint (``--tracking-uri`` overrides it), not the notebook's
MLFLOW_TRACKING_URI, which names an internal host this pod cannot resolve.

Refused, so they never reach MLflow:
- smoke runs (``smoke_tiles_per_slide`` set, or "smoke" in the path);
- a model set without all 5 finished folds, or with fewer epoch files than
  configured;
- a set whose report entry (best epoch, threshold, folds) does not match its
  run dir, i.e. the report was not built from these runs;
- a param value longer than the server limit (500 chars);
- a set already logged with the same key (tag ``huncrc.log_key``), unless
  ``--allow-duplicate``.

``--check-mlflow`` and ``--dry-run`` never write to MLflow; ``--dry-run`` never
imports it. Credentials are never printed, only whether they are set.
test_final is never read.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.data import huncrc_features, huncrc_labels  # noqa: E402
from ml.data.huncrc_labels import LabelConfig  # noqa: E402
from ml.huncrc_mlflow_auth import (  # noqa: E402
    AUTH_VARS,
    PUBLIC_MLFLOW,
    AuthMissingError,
)
from ml.huncrc_mlflow_auth import mask as _mask  # noqa: E402
from ml.huncrc_mlflow_auth import redact_uri as _redact_uri  # noqa: E402
from ml.huncrc_mlflow_auth import setup_tracking as _setup_tracking  # noqa: E402


EXPERIMENT_ID = "168"
EXPERIMENT_NAME = "Ulcerative Colitis Dysplasia"
BRANCH = "feature/huncrc-tile-classifier"
N_FOLDS = 5
FINAL_SETS = ["A-unw", "A-bal", "B", "C", "D", "E"]
MAX_PARAM_LEN = 500  # MLflow 2.x server limit

OLD_RULES = LabelConfig()
NEW_RULES = LabelConfig(within_slide_negatives=True, exclude_carcinoma=True)


@dataclass(frozen=True)
class SetSpec:
    subdir: str  # directory name inside the runs dir
    head: str
    weighting: str
    rules: LabelConfig
    step: str
    title: str


SETS = {
    "A-unw": SetSpec("a", "linear", "none", OLD_RULES, "4a", "linear, Step-4a labels, unweighted"),
    "A-bal": SetSpec("b", "linear", "balanced", OLD_RULES, "4a", "linear, Step-4a labels, class-balanced"),
    "B": SetSpec("B", "linear", "none", NEW_RULES, "8", "linear, within-slide negatives, unweighted"),
    "C": SetSpec("C", "linear", "balanced", NEW_RULES, "8", "linear, within-slide negatives, class-balanced"),
    "D": SetSpec("D", "mlp", "none", NEW_RULES, "9", "MLP, within-slide negatives, unweighted"),
    "E": SetSpec("E", "mlp", "balanced", NEW_RULES, "9", "MLP, within-slide negatives, class-balanced"),
}  # fmt: skip


def head_arch(head: str, dim: int) -> str:
    return {
        "linear": f"Dropout(0.5) -> Linear({dim}, 1)  [EmbeddingClassifier]",
        "mlp": f"Dropout(0.5) -> {dim}-512-256 ReLU -> Linear(256, 1)  "
        "[MlpEmbeddingClassifier]",
    }[head]


#: where a run's embedding model and dimension come from (param embedding_source)
EMB_RUN_CONFIG = "run-config"
EMB_CACHE_META = "train-cache-meta"
EMB_LEGACY_4A = "legacy-step4a"
EMBEDDING_BASIS = {
    EMB_RUN_CONFIG: "embedding_model / embedding_dim recorded by within_slide_cv.py "
    "in runs/run_config.json",
    EMB_CACHE_META: "meta.json of the train cache named in runs/run_config.json "
    "(the run predates the run_config record)",
    EMB_LEGACY_4A: "not recorded: Step 4a (2026-11-19) wrote no run_config; "
    "prov-gigapath 1536-d was the only embedding on the share then",
}
VAL_KEYS = ["auroc", "auprc", "sensitivity", "specificity", "f1"]
EPOCH_KEYS = ["train_loss", "val_auroc", "val_auprc", "seconds"]
ALLOWED_ARTIFACTS = {".txt", ".json", ".csv"}

STATIC_PARAMS = {
    "embedding_mpp": "0.49",
    "tiling": "224 px tiles, level 2 (downsample 4), stride 112",
    "tile_label": "coverage = max(LG, HG) over the central half (112 px) of the tile",
    "split": "slides.parquet: train 140 / test_preliminary 30 / test_final 30 slides "
    "(test_final never read)",
    "folds": "5 slide-level folds of the 140 train slides (28 each) from slides.parquet; "
    "fold k = validation fold k, trained on the other 4; metric step = fold index",
    "optimizer": "AdamW",
    "loss": "BCEWithLogitsLoss (class-balanced variants: per-tile class weights)",
    "scaler": "per-feature standardisation fitted on the training folds only",
    "model_selection": "best epoch = max validation AUPRC",
    "threshold_rule": "Youden J on the validation fold",
    "seed": "fold index (0-4)",
    "null_reference": "validation only: label-permutation AUROC/AUPRC per fold "
    "(the report has no test_preliminary null)",
    "test_preliminary_use": "model/config comparison only; no tuning",
}


class RefuseError(RuntimeError):
    """The input is not a final, complete, consistent model set."""


@dataclass
class RunPlan:
    name: str
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    #: headline values (mean over folds, sd, CI), logged at step 0
    metrics: dict[str, float] = field(default_factory=dict)
    #: name -> [(step, value)]: per-fold values (step = fold) and epoch curves
    series: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    #: artifact file name -> file content (built in memory, written at log time)
    artifacts: dict[str, str] = field(default_factory=dict)
    skipped_metrics: list[str] = field(default_factory=list)
    start_ms: int | None = None
    end_ms: int | None = None


# ------------------------------------------------------------------ helpers


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _put(plan: RunPlan, key: str, value: Any) -> None:
    """Add a finite headline metric; NaN/None is skipped and listed."""
    if not _finite(value):
        plan.skipped_metrics.append(key)
        return
    plan.metrics[key] = float(value)


def _put_folds(plan: RunPlan, key: str, values: list[Any], folds: list[int]) -> None:
    """``key_fold`` with step = fold; NaN folds are left out and listed."""
    pts = [(k, float(v)) for k, v in zip(folds, values, strict=True) if _finite(v)]
    if len(pts) < len(folds):
        bad = [k for k, v in zip(folds, values, strict=True) if not _finite(v)]
        plan.skipped_metrics.append(f"{key}_fold@{','.join(map(str, bad))}")
    if pts:
        plan.series[f"{key}_fold"] = pts


def _csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    cols = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _mean_sd(v: list[Any]) -> tuple[float, float]:
    a = np.asarray([float(x) if _finite(x) else np.nan for x in v], dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float("nan")


# How much a recorded commit proves about the code that ran (git_commit_source):
JOB_CLONE = "job-clone"  # fresh clone of a pushed commit in the pod: exact
RUN_LOG_HEAD = "run-log-head"  # HEAD of the notebook checkout at run start
UNRECORDED = "unrecorded"  # the run logged no commit
RETROSPECTIVE = "retrospective"  # assigned afterwards (--git-commit)

GIT_DIRTY_BASIS = {
    JOB_CLONE: "fresh git clone of the pushed commit in the job pod; the tree "
    "cannot differ from the commit",
    RUN_LOG_HEAD: "the run log checked only unstaged changes to tracked files under "
    "ml/ and scripts/ at run start (git diff --quiet); staged, untracked and other "
    "files were not checked, so a clean tree is not established",
    UNRECORDED: "the run recorded no commit and no tree state",
    RETROSPECTIVE: "the commit was assigned after the run; the run recorded no tree "
    "state",
}


def run_embedding(
    name: str, set_dir: Path, source_map: dict[str, str] | None = None
) -> tuple[str, int, str]:
    """(model, dim, source) of a model set's embedding, from the run itself.

    run_config.json record, else the meta.json of the train cache it names,
    else (Step 4a only) the labelled legacy value. The dimension is
    cross-checked against every saved per-fold scaler.
    """
    rc_path = set_dir.parent / "run_config.json"
    rc = _read_json(rc_path) if rc_path.is_file() else {}
    if rc.get("embedding_model") and rc.get("embedding_dim"):
        model, dim, src = (
            str(rc["embedding_model"]),
            int(rc["embedding_dim"]),
            EMB_RUN_CONFIG,
        )
    elif rc.get("train_cache"):
        meta = Path(map_source(str(rc["train_cache"]), source_map or {})) / "meta.json"
        if not meta.is_file():
            raise RefuseError(
                f"{name}: {rc_path} records no embedding and its train cache "
                f"{meta.parent} is gone; pass --source-map OLD=NEW if it moved"
            )
        model, dim = huncrc_features.cache_embedding(_read_json(meta))
        src = EMB_CACHE_META
    elif SETS[name].step == "4a":
        model, dim, src = (huncrc_features.DEFAULT_MODEL,
                           huncrc_features.DEFAULT_DIM, EMB_LEGACY_4A)  # fmt: skip
    else:
        raise RefuseError(f"{name}: no embedding recorded ({rc_path} missing or empty)")
    for sc in sorted(set_dir.glob("fold*/scaler.npz")):
        n = len(np.load(sc)["mean"])
        if n != dim:
            raise RefuseError(
                f"{name}: {sc} has {n} features, the run says {model} {dim}-d"
            )
    return model, dim, src


def code_provenance(set_dir: Path) -> dict[str, str]:
    """Git commit, how it was recorded, tree state and origin, from the run log.

    run.log next to the runs dir (Steps 8-9: ``<out>/run.log``, ``<out>/runs``),
    else train.log in the runs dir (Step 4a). Only a fresh clone in a job pod
    proves a clean tree; a notebook run's ``git=<sha>`` without ``+dirty``
    leaves git_dirty "unknown" (see GIT_DIRTY_BASIS).
    """
    runs_dir = set_dir.parent
    logs = [runs_dir.parent / "run.log", runs_dir / "run.log", runs_dir / "train.log"]
    for log in logs:
        if not log.is_file():
            continue
        text = log.read_text(errors="replace")
        origin = "cluster-job" if "[job] " in text else "jupyter"
        m = re.search(r"\[job\] commit ([0-9a-f]{40})", text)
        g = re.search(r"\bgit=([0-9a-f]{7,40})(\+dirty)?", text)
        if m:
            sha, source, dirty = m.group(1), JOB_CLONE, "false"
        elif g:
            sha, source = g.group(1), RUN_LOG_HEAD
            dirty = "true" if g.group(2) else "unknown"
        else:
            sha, source, dirty = "unrecorded", UNRECORDED, "unknown"
        return {"git_commit": sha, "git_commit_source": source, "git_dirty": dirty,
                "origin": origin, "log": str(log)}  # fmt: skip
    return {"git_commit": "unrecorded", "git_commit_source": UNRECORDED,
            "git_dirty": "unknown", "origin": "unknown", "log": ""}  # fmt: skip


def map_source(path: str, source_map: dict[str, str]) -> str:
    """Rewrite the longest matching path prefix (OLD=NEW), whole components only."""
    best = ""
    for old in source_map:
        o = old.rstrip("/")
        if (path == o or path.startswith(o + "/")) and len(o) > len(best):
            best = o
    if not best:
        return path
    return source_map[best].rstrip("/") + path[len(best) :]


def resolve_set_dir(
    name: str, report: dict[str, Any], source_map: dict[str, str]
) -> tuple[Path, str]:
    """(dir to read, path recorded in the report)."""
    src = report.get("sources", {}).get(name)
    if not src:
        raise RefuseError(f"{name}: the report has no source dir for it")
    p = Path(map_source(src, source_map))
    if not p.is_dir():
        raise RefuseError(f"{name}: source {p} does not exist (report: {src})")
    if p.name != SETS[name].subdir:
        raise RefuseError(f"{name}: source {p} is not a '{SETS[name].subdir}/' dir")
    return p, src


def _check_not_smoke(name: str, set_dir: Path) -> None:
    # <out>/runs/<set> (Steps 8-9) or <runs dir>/<set> (Step 4a)
    own = [set_dir, set_dir.parent] + (
        [set_dir.parent.parent] if set_dir.parent.name == "runs" else []
    )
    if any("smoke" in d.name.lower() for d in own):
        raise RefuseError(f"{name}: {set_dir} looks like a smoke run")
    rc = set_dir.parent / "run_config.json"
    if rc.is_file() and _read_json(rc).get("smoke_tiles_per_slide") not in (
        None,
        "None",
    ):
        raise RefuseError(f"{name}: {rc} has smoke_tiles_per_slide set")


def _one(name: str, key: str, values: set[str]) -> str:
    if len(values) != 1:
        raise RefuseError(f"{name}: {key} differs between folds: {sorted(values)}")
    return values.pop()


# ------------------------------------------------------------------ assembly


def build_plan(
    name: str,
    report: dict[str, Any],
    set_dir: Path,
    report_path: Path,
    table_text: str | None,
    git_override: str | None = None,
    reported_dir: str | None = None,
    source_map: dict[str, str] | None = None,
) -> RunPlan:
    spec = SETS[name]
    if name not in report.get("model_sets", {}):
        raise RefuseError(f"{name}: not in {report_path}")
    R = report["model_sets"][name]
    _check_not_smoke(name, set_dir)
    folds = list(R["folds"])
    if folds != list(range(N_FOLDS)):
        raise RefuseError(f"{name}: report has folds {folds}, need all {N_FOLDS}")

    emb_model, emb_dim, emb_src = run_embedding(name, set_dir, source_map)
    prov = code_provenance(set_dir)
    recorded = prov["git_commit"]
    if git_override:
        prov["git_commit"], prov["git_dirty"] = git_override, "unknown"
        prov["git_commit_source"] = RETROSPECTIVE
    rules = spec.rules
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()[:12]
    log_key = f"{name}|{prov['git_commit'][:12]}|{report_sha}"

    # ---- per fold: saved run files, cross-checked against the report
    dones, counts_l, epochs_l, mtimes = [], [], [], []
    for k in folds:
        fd = set_dir / f"fold{k}"
        if not (fd / "done.json").is_file():
            raise RefuseError(f"{name}: {fd} has no done.json (unfinished)")
        done = _read_json(fd / "done.json")
        cfg = done["config"]
        if done.get("head", spec.head) != spec.head:
            raise RefuseError(f"{name} fold{k}: head {done.get('head')} != {spec.head}")
        counts = (
            _read_json(fd / "counts.json") if (fd / "counts.json").is_file() else None
        )
        if counts and counts.get("smoke_tiles_per_slide") is not None:
            raise RefuseError(f"{name} fold{k}: smoke run (smoke_tiles_per_slide set)")
        epochs = [_read_json(fd / f"epoch_{e:02d}.json") for e in range(done["epochs"])
                  if (fd / f"epoch_{e:02d}.json").is_file()]  # fmt: skip
        if len(epochs) != done["epochs"] or done["epochs"] != cfg["epochs"]:
            raise RefuseError(
                f"{name} fold{k}: {len(epochs)} epoch files, {done['epochs']} configured"
            )
        if int(cfg["seed"]) != k:
            raise RefuseError(f"{name} fold{k}: seed {cfg['seed']} != fold index")
        v = R["val"][f"fold{k}"]
        if v["best_epoch"] != done["best_epoch"] or not math.isclose(
            v["threshold"], done["threshold"], rel_tol=0, abs_tol=1e-9
        ):
            raise RefuseError(f"{name} fold{k}: report and {fd}/done.json disagree")
        dones.append(done)
        counts_l.append(counts)
        epochs_l.append(epochs)
        mtimes.append((fd / "done.json").stat().st_mtime)

    cfgs = [d["config"] for d in dones]
    params: dict[str, str] = {
        "model_set": name,
        "head": spec.head,
        "head_arch": head_arch(spec.head, emb_dim),
        "embedding_model": emb_model,
        "embedding_dim": str(emb_dim),
        "embedding_source": emb_src,
        "embedding_source_basis": EMBEDDING_BASIS[emb_src],
        "class_weighting": spec.weighting,
        "label_rules_fingerprint": huncrc_labels.rules_fingerprint(rules),
        "label_rules_version": str(huncrc_labels.LABEL_RULES_VERSION),
        "feature_cache_version": str(huncrc_features.CACHE_VERSION),
        "within_slide_negatives": str(rules.within_slide_negatives).lower(),
        "exclude_carcinoma": str(rules.exclude_carcinoma).lower(),
        "excluded_slides": " ".join(sorted(rules.exclude_slides)),
        "exclude_serrated": str(rules.exclude_serrated).lower(),
        "safe_categories": " ".join(sorted(rules.safe_categories)),
        "min_coverage": f"{rules.min_coverage:g}",
        "epochs": _one(name, "epochs", {str(c["epochs"]) for c in cfgs}),
        "lr": _one(name, "lr", {f"{c['lr']:g}" for c in cfgs}),
        "batch_size": _one(name, "batch_size", {str(c["batch_size"]) for c in cfgs}),
        "dropout": _one(name, "dropout", {f"{c['dropout']:g}" for c in cfgs}),
        "weight_decay": _one(name, "weight_decay", {f"{c['weight_decay']:g}" for c in cfgs}),
        "neg_cap_per_slide": _one(name, "neg_cap", {str(c["neg_cap"]) if c else "none"
                                                    for c in counts_l}),
        "class_weights_per_fold": "none" if all(d["class_weights"] is None for d in dones)
        else "; ".join("fold{}: neg={:.6f} pos={:.6f}".format(k, *d["class_weights"])
                       for k, d in zip(folds, dones, strict=True)),
        "git_commit": prov["git_commit"],
        "git_commit_source": prov["git_commit_source"],
        "git_dirty": prov["git_dirty"],
        "git_dirty_basis": GIT_DIRTY_BASIS[prov["git_commit_source"]],
        "runs_dir": reported_dir or str(set_dir),
        **STATIC_PARAMS,
    }  # fmt: skip
    if reported_dir and reported_dir != str(set_dir):
        params["runs_dir_copy"] = str(set_dir)
    if prov["git_commit_source"] == RETROSPECTIVE:
        params["git_commit_recorded"] = recorded
    if spec.step == "4a":
        params["git_note"] = (
            "Step 4a ran 2026-11-19, before its code was committed; "
            "scripts/huncrc/linear_cv.py was first committed in a9e6511 (2026-11-21)"
        )
    for key, val in R["prelim"].get("n", {}).items():
        params[f"prelim_n_{key}"] = (
            " ".join(map(str, val)) if isinstance(val, list) else str(val)
        )
    long = [k for k, val in params.items() if len(val) > MAX_PARAM_LEN]
    if long:
        raise RefuseError(f"{name}: params longer than {MAX_PARAM_LEN} chars: {long}")

    tags = {
        "step": spec.step,
        "branch": BRANCH,
        "origin": prov["origin"],
        "git_commit_source": prov["git_commit_source"],
        "model_set": name,
        "retrospective": "true",
        "smoke": "false",
        "huncrc.log_key": log_key,
        "mlflow.note.content": (
            f"HunCRC Step {spec.step}, model set {name}: {spec.title}. 5-fold CV on "
            f"frozen {emb_model} embeddings; logged retrospectively from saved runs "
            "(no retraining). Metric X = mean over folds, X_fold = per fold "
            "(step = fold), X_sd = sd over folds."
        ),
    }
    if prov["git_commit_source"] in (JOB_CLONE, RUN_LOG_HEAD):  # recorded by the run
        tags["mlflow.source.git.commit"] = prov["git_commit"]

    plan = RunPlan(name=name, tags=tags, params=params)

    # ---- validation metrics: mean (headline), per fold, sd
    V = [R["val"][f"fold{k}"] for k in folds]
    val_cols: dict[str, list[Any]] = {}
    for part in ("new", "old"):
        for key in [*VAL_KEYS, "slide_auroc", "n_tiles", "prevalence"]:
            val_cols[f"val_{part}_{key}"] = [v[part].get(key) for v in V]
        for key in VAL_KEYS:
            for j, side in enumerate(("ci_low", "ci_high")):
                col = [(v[f"{part}_ci"].get(key) or [None, None])[j] for v in V]
                if any(_finite(x) for x in col):
                    val_cols[f"val_{part}_{key}_{side}"] = col
    val_cols["val_threshold"] = [v["threshold"] for v in V]
    val_cols["val_best_epoch"] = [v["best_epoch"] for v in V]
    val_cols["val_null_auroc"] = [v["null_auroc"][0] for v in V]
    val_cols["val_null_auroc_perm_sd"] = [v["null_auroc"][1] for v in V]
    val_cols["val_null_auprc"] = [v["null_auprc"][0] for v in V]
    val_cols["val_null_auprc_perm_sd"] = [v["null_auprc"][1] for v in V]
    for key, col in val_cols.items():
        mean, sd = _mean_sd(col)
        _put(plan, key, mean)
        _put(plan, f"{key}_sd", sd)
        _put_folds(plan, key, col, folds)

    # ---- test_preliminary: report summary (mean, sd, bootstrap CI), per fold
    per_fold = R["prelim"]["per_fold"]
    for key, s in R["prelim"]["summary"].items():
        _put(plan, f"prelim_{key}", s["mean"])
        _put(plan, f"prelim_{key}_sd", s["sd"])
        if "ci" in s:
            _put(plan, f"prelim_{key}_ci_low", s["ci"][0])
            _put(plan, f"prelim_{key}_ci_high", s["ci"][1])
        _put_folds(plan, f"prelim_{key}", [pf.get(key) for pf in per_fold], folds)

    # ---- per-epoch curves, step = epoch (1-based)
    for k, epochs in zip(folds, epochs_l, strict=True):
        for key in EPOCH_KEYS:
            plan.series[f"fold{k}_epoch_{key}"] = [
                (int(e["epoch"]), float(e[key])) for e in epochs
            ]

    # ---- artifacts (small text summaries only)
    ps = R["prelim"].get("per_slide_ws_auroc", [])
    slides = sorted({sl for d in ps for sl in d})
    plan.artifacts["per_slide_ws_auroc.csv"] = _csv(
        [{"slide": sl, **{f"fold{k}": ps[i].get(sl, "") for i, k in enumerate(folds)},
          "mean": float(np.nanmean([ps[i][sl] for i in range(len(ps)) if sl in ps[i]]))}
         for sl in slides]
    )  # fmt: skip
    plan.artifacts["val_per_fold.csv"] = _csv(
        [
            {"fold": k, **{c: val_cols[c][i] for c in val_cols}}
            for i, k in enumerate(folds)
        ]
    )
    plan.artifacts["prelim_per_fold.csv"] = _csv(
        [{"fold": k, **per_fold[i]} for i, k in enumerate(folds)]
    )
    plan.artifacts["epochs.csv"] = _csv(
        [{"fold": k, **e} for k, ep in zip(folds, epochs_l, strict=True) for e in ep]
    )
    plan.artifacts["folds.json"] = json.dumps(
        {f"fold{k}": {"done": d, "counts": c}
         for k, d, c in zip(folds, dones, counts_l, strict=True)}, indent=2
    )  # fmt: skip
    plan.artifacts["report.json"] = report_path.read_text()
    if table_text is not None:
        plan.artifacts["table.txt"] = table_text
    bad = [a for a in plan.artifacts if Path(a).suffix not in ALLOWED_ARTIFACTS]
    assert not bad, f"disallowed artifacts {bad}"

    plan.end_ms = int(max(mtimes) * 1000)
    plan.start_ms = int(
        min(mt - sum(float(e["seconds"]) for e in ep)
            for mt, ep in zip(mtimes, epochs_l, strict=True)) * 1000
    )  # fmt: skip
    return plan


def build_plans(
    report_path: Path,
    sets: list[str],
    table_path: Path | None,
    git_overrides: dict[str, str],
    source_map: dict[str, str] | None = None,
) -> list[RunPlan]:
    report = _read_json(report_path)
    missing = [n for n in sets if n not in report.get("model_sets", {})]
    if missing:
        raise RefuseError(f"{', '.join(missing)}: not in {report_path}")
    table_text = table_path.read_text() if table_path and table_path.is_file() else None
    plans = []
    for n in sets:
        set_dir, reported = resolve_set_dir(n, report, source_map or {})
        plans.append(
            build_plan(
                n,
                report,
                set_dir,
                report_path,
                table_text,
                git_overrides.get(n),
                reported,
                source_map,
            )
        )
    return plans


# ------------------------------------------------------------------ output


def describe(plans: list[RunPlan], experiment_id: str) -> str:
    L = [
        f"DRY RUN: nothing is sent to MLflow. Experiment {experiment_id} "
        f"({EXPERIMENT_NAME!r}).",
        f"{len(plans)} runs, no nested runs: {', '.join(p.name for p in plans)}",
        "Metric X = mean over folds (step 0); X_fold = per fold (step = fold); "
        "X_sd = sd over folds; fold<k>_epoch_* step = epoch.\n",
    ]
    for p in plans:
        n_pts = sum(len(s) for s in p.series.values())
        L.append(f"RUN {p.name!r}  start={p.start_ms} end={p.end_ms}")
        L.append(f"  tags ({len(p.tags)}):")
        L.extend(f"    {k} = {v}" for k, v in sorted(p.tags.items()))
        L.append(f"  params ({len(p.params)}):")
        L.extend(f"    {k} = {v}" for k, v in sorted(p.params.items()))
        L.append(f"  metrics, headline ({len(p.metrics)}):")
        L.extend(f"    {k} = {v:.6g}" for k, v in sorted(p.metrics.items()))
        L.append(f"  metrics, stepped ({len(p.series)} names, {n_pts} points):")
        L.extend(
            f"    {k}: " + " ".join(f"{s}:{v:.4g}" for s, v in c)
            for k, c in sorted(p.series.items())
        )
        if p.skipped_metrics:
            L.append(f"  skipped (NaN): {', '.join(p.skipped_metrics)}")
        L.append(f"  artifacts ({len(p.artifacts)}):")
        L.extend(
            f"    {k} ({len(v.encode())} bytes)" for k, v in sorted(p.artifacts.items())
        )
        L.append("")
    return "\n".join(L)


def setup_tracking(
    uri: str,
    login: bool,
    ask: Any = input,
    ask_hidden: Any = getpass.getpass,
) -> str:
    """ml.huncrc_tracking.setup_tracking, with a CLI exit when auth is missing."""
    try:
        return _setup_tracking(uri, login, ask=ask, ask_hidden=ask_hidden)
    except AuthMissingError as e:
        raise SystemExit(
            f"MLflow authentication is missing ({e}): run with --login in a Jupyter "
            "terminal (or set MLFLOW_TRACKING_TOKEN). Nothing done."
        ) from None


def check_mlflow(experiment_id: str, auth_mode: str = "") -> int:
    """Read-only: where, which credentials, experiment name and artifact location."""
    from mlflow.tracking import MlflowClient

    from ml._mlflow_compat import apply_mlflow_compat_patch

    apply_mlflow_compat_patch()
    print(
        f"MLFLOW_TRACKING_URI = {_redact_uri(os.environ.get('MLFLOW_TRACKING_URI', ''))}"
    )
    if auth_mode:
        print(f"credentials: {auth_mode}")
    for var in AUTH_VARS:
        print(f"{var}: {'set' if os.environ.get(var) else 'not set'}")
    try:
        exp = MlflowClient().get_experiment(experiment_id)
    except Exception as e:  # report and stop; never retry with other credentials
        msg = _mask(str(e))
        print(
            f"get_experiment({experiment_id}) FAILED: {type(e).__name__}: {msg[:600]}"
        )
        return 2
    print(f"experiment {exp.experiment_id}: name={exp.name!r}")
    print(f"  artifact_location = {exp.artifact_location}")
    print(f"  lifecycle_stage   = {exp.lifecycle_stage}")
    if exp.name != EXPERIMENT_NAME:
        print(f"  NAME MISMATCH: expected {EXPERIMENT_NAME!r}")
        return 3
    print("read access OK (write permission is not tested: that would need a write)")
    return 0


def log_plans(plans: list[RunPlan], experiment_id: str, allow_duplicate: bool) -> None:
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    from ml._mlflow_compat import apply_mlflow_compat_patch

    apply_mlflow_compat_patch()
    client = MlflowClient()
    exp = client.get_experiment(experiment_id)
    if exp is None or exp.name != EXPERIMENT_NAME:
        raise SystemExit(
            f"experiment {experiment_id} is {exp and exp.name!r}, expected {EXPERIMENT_NAME!r}"
        )
    for p in plans:
        key = p.tags["huncrc.log_key"]
        found = client.search_runs([experiment_id], f"tags.`huncrc.log_key` = '{key}'")
        if found and not allow_duplicate:
            raise SystemExit(f"{p.name}: already logged as run {found[0].info.run_id}")

    for p in plans:
        run = client.create_run(
            experiment_id, start_time=p.start_ms, tags=p.tags, run_name=p.name
        )
        rid = run.info.run_id
        try:
            ts = p.end_ms or 0
            params = [Param(k, v) for k, v in p.params.items()]
            metrics = [Metric(k, v, ts, 0) for k, v in p.metrics.items()]
            metrics += [Metric(k, v, ts, s) for k, c in p.series.items() for s, v in c]
            for i in range(0, max(len(params), len(metrics)), 100):
                client.log_batch(
                    rid, metrics=metrics[i : i + 100], params=params[i : i + 100]
                )
            with tempfile.TemporaryDirectory() as tmp:
                for name, text in p.artifacts.items():
                    f = Path(tmp) / name
                    f.write_text(text)
                    client.log_artifact(rid, str(f))
            client.set_terminated(rid, "FINISHED", end_time=p.end_ms)
        except Exception:
            client.set_terminated(rid, "FAILED")
            raise
        print(f"logged {p.name!r}: run {rid}", flush=True)


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for s in items:
        if "=" not in s:
            raise SystemExit(f"{what}: expected OLD=NEW, got {s!r}")
        k, v = s.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", type=Path, help="combined report.json")
    ap.add_argument(
        "--table", type=Path, help="table.txt (default: next to the report)"
    )
    ap.add_argument("--source-map", nargs="*", default=[], metavar="OLD=NEW",
                    help="rewrite a path prefix of the report's sources (runs copied to the share)")  # fmt: skip
    ap.add_argument("--sets", nargs="+", default=FINAL_SETS, choices=FINAL_SETS)
    ap.add_argument("--git-commit", nargs="*", default=[], metavar="SET=SHA",
                    help="override the code commit of a set (e.g. Step 4a, which logged none)")  # fmt: skip
    ap.add_argument("--experiment-id", default=EXPERIMENT_ID)
    ap.add_argument("--tracking-uri", default=PUBLIC_MLFLOW,
                    help=f"MLflow server (default {PUBLIC_MLFLOW})")  # fmt: skip
    ap.add_argument("--login", action="store_true",
                    help="prompt for login email and API access token (hidden)")  # fmt: skip
    ap.add_argument("--allow-duplicate", action="store_true")
    ap.add_argument("--check-mlflow", action="store_true",
                    help="read-only check of the server and experiment, then exit")  # fmt: skip
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-run-out", type=Path, help="also write the dry-run text here")
    a = ap.parse_args(argv)
    if a.login and a.dry_run:
        ap.error("--dry-run never contacts MLflow; drop --login")
    if a.check_mlflow:
        mode = setup_tracking(a.tracking_uri, a.login)
        raise SystemExit(check_mlflow(a.experiment_id, mode))
    if a.report is None:
        ap.error("--report is required")
    overrides = _pairs(a.git_commit, "--git-commit")
    smap = _pairs(a.source_map, "--source-map")
    table = a.table or a.report.with_name("table.txt")
    try:
        plans = build_plans(a.report, a.sets, table, overrides, smap)
    except RefuseError as e:
        raise SystemExit(f"refused: {e}") from None
    if a.dry_run:
        text = describe(plans, a.experiment_id)
        print(text)
        if a.dry_run_out:
            a.dry_run_out.parent.mkdir(parents=True, exist_ok=True)
            a.dry_run_out.write_text(text + "\n")
        return
    mode = setup_tracking(a.tracking_uri, a.login)
    print(f"MLflow {_redact_uri(a.tracking_uri)}; credentials: {mode}", flush=True)
    try:
        log_plans(plans, a.experiment_id, a.allow_duplicate)
    except Exception as e:
        raise SystemExit(
            f"logging failed: {type(e).__name__}: {_mask(str(e))[:600]}"
        ) from None


if __name__ == "__main__":
    main()
