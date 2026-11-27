"""Hydra entry point for the HunCRC tile classifier (Steps 8-9 protocol).

    python -m ml.huncrc_cv experiment=huncrc/D data.out=/mnt/projects/.../d_run
    python -m ml.huncrc_cv experiment=huncrc/D data.out=... mlflow.login=true   # Jupyter
    python -m ml.huncrc_cv experiment=huncrc/D data.out=... mlflow.enabled=false  # job
    python -m ml.huncrc_cv experiment=huncrc/B data.out=... dry_run=true

A thin layer: it builds (or reuses) the two feature caches exactly as
scripts/huncrc/run_cv.sh does, then calls ``within_slide_cv.main`` with the
same arguments the argparse CLI takes, so training, resume and outputs are
identical; that CLI and run_cv.sh stay as they are.

House convention: ``@hydra.main`` over ``@autolog`` with ``main(config,
logger)``. The logger is :class:`ml.huncrc_tracking.OptionalMLFlowLogger`,
which never raises, so a run without MLflow credentials (every cluster job)
or with an unreachable server just writes to disk. With a usable server the
run gets the per-epoch curves live and, after the report, the same params,
metrics and artifacts as scripts/huncrc/log_to_mlflow.py would log (same
names; same ``huncrc.log_key``, so a later retrospective log refuses the
duplicate).

Before anything reaches MLflow the config is checked. Refused (unsafe): a
credential-like key or an ``oc.env`` interpolation (``@autolog`` uploads the
config), any value naming test_final, and a smoke run aimed at the main
experiment. Also refused, because the code cannot honour them and the run
would be logged with values it did not use: a head/class weighting that is
not the chosen model set, and training hyperparameters that
within_slide_cv.py does not take yet. Label rules other than the Step-8 ones
are allowed and trained on, with a loud warning naming what differs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog

from ml.huncrc_tracking import AUTH_VARS, OptionalMLFlowLogger, find_secrets


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from ml.data.huncrc_labels import LabelConfig


REPO = Path(__file__).resolve().parents[1]
MAIN_EXPERIMENT = "Ulcerative Colitis Dysplasia"


def _script(name: str) -> ModuleType:
    """scripts/huncrc/<name>.py (scripts/ is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / "huncrc" / f"{name}.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ checks


def _items(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """(dotted key, raw value) pairs of a config, interpolations unresolved."""
    out: list[tuple[str, Any]] = []
    if OmegaConf.is_dict(node):
        for k in node:
            out += _items(node._get_node(k), f"{prefix}{k}.")
        return out
    if OmegaConf.is_list(node):
        for i in range(len(node)):
            out += _items(node._get_node(i), f"{prefix}{i}.")
        return out
    return [(prefix.rstrip("."), node._value())]


def label_config(config: DictConfig) -> LabelConfig:
    from ml.data.huncrc_labels import LabelConfig

    lab = config.labels
    return LabelConfig(
        min_coverage=float(lab.min_coverage),
        exclude_slides=frozenset(map(str, lab.exclude_slides)),
        safe_categories=frozenset(map(str, lab.safe_categories)),
        exclude_serrated=bool(lab.exclude_serrated),
        within_slide_negatives=bool(lab.within_slide_negatives),
        exclude_carcinoma=bool(lab.exclude_carcinoma),
    )


def label_rules_diff(config: DictConfig) -> dict[str, tuple[Any, Any]]:
    """Fields of the configured label rules that differ from the Step-8 rules."""
    from ml.data.huncrc_labels import rules_diff

    return rules_diff(label_config(config), _script("within_slide_cv").NEW_RULES)


def label_rules_warning(config: DictConfig) -> str | None:
    diff = label_rules_diff(config)
    if not diff:
        return None
    lines = [f"  {k}: {v[0]}   (Step 8: {v[1]})" for k, v in diff.items()]
    bar = "!" * 72
    return "\n".join([
        bar,
        "WARNING: label rules differ from the Step-8 rules of the final model sets",
        *lines,
        "This run trains on these rules (new cache key) and is NOT comparable to",
        f"the logged {config.model_set} run as the same model set.",
        bar,
    ])  # fmt: skip


def check_config(config: DictConfig, overrides: list[str] | None = None) -> str:
    """Refuse a config that is unsafe or that the code cannot honour.

    Returns the within_slide_cv variant (B C D E). Label rules are not checked
    here; see :func:`label_rules_warning`.
    """
    items = _items(config)
    bad = find_secrets(items)
    parsed = [o.lstrip("+~").partition("=") for o in overrides or []]
    for key, _, value in parsed:
        bad += find_secrets([(f"override {key}", value)])
    if bad:
        raise SystemExit(
            "config refused (it is uploaded to MLflow):\n  " + "\n  ".join(bad)
        )
    final = [k for k, v in items if "test_final" in str(v)]
    final += [f"override {k}" for k, _, v in parsed if "test_final" in f"{k}={v}"]
    if final:
        raise SystemExit(f"config refused: test_final is never read ({final})")
    label_config(config)  # valid label rules (raises on e.g. min_coverage > 1)

    wcv = _script("within_slide_cv")
    head, weighting = config.head.kind, config.training.class_weighting
    by_spec = {spec: name for name, spec in wcv.VARIANTS.items()}
    variant = by_spec.get((head, None if weighting == "none" else weighting))
    if variant is None:
        raise SystemExit(f"no model set has head={head} class_weighting={weighting}")
    if variant != config.model_set:
        raise SystemExit(
            f"model_set={config.model_set} but head={head} "
            f"class_weighting={weighting} is model set {variant}"
        )

    from ml.huncrc_train import TrainConfig

    t, ref = config.training, TrainConfig()
    fixed = {"lr": ref.lr, "batch_size": ref.batch_size,
             "weight_decay": ref.weight_decay}  # fmt: skip
    diff: dict[str, tuple[Any, Any]] = {
        k: (t[k], v) for k, v in fixed.items() if float(t[k]) != float(v)
    }
    if float(config.head.dropout) != ref.dropout:
        diff["dropout"] = (config.head.dropout, ref.dropout)
    if head == "mlp" and list(config.head.hidden) != [512, 256]:
        diff["hidden"] = (list(config.head.hidden), [512, 256])
    if diff:
        raise SystemExit(
            f"within_slide_cv.py does not take these yet (config, code): {diff}"
        )

    if (
        config.training.smoke_tiles_per_slide is not None
        and config.mlflow.enabled
        and config.metadata.experiment_name == MAIN_EXPERIMENT
    ):
        raise SystemExit(
            "smoke run: set metadata.experiment_name to another experiment, or "
            "mlflow.enabled=false; smoke runs never go to the main experiment"
        )
    return str(variant)


def checked(func: Callable[[DictConfig], None]) -> Callable[[DictConfig], None]:
    """Run :func:`check_config` before @autolog uploads anything."""

    @wraps(func)
    def wrapper(config: DictConfig) -> None:
        try:
            overrides = list(HydraConfig.get().overrides.task)
        except ValueError:
            overrides = []
        check_config(config, overrides)
        if config.dry_run:  # a dry run never creates an MLflow run
            config.mlflow.enabled = False
        return func(config)

    return wrapper


# ------------------------------------------------------------------ calls


def build_calls(config: DictConfig, variant: str, train_cache: str, prelim_cache: str
                ) -> tuple[list[str], list[str] | None]:  # fmt: skip
    """within_slide_cv.py argv for train and report, as run_cv.sh builds them."""
    d, t, e = config.data, config.training, config.evaluation
    common = ["--train-cache", train_cache, "--prelim-all-cache", prelim_cache,
              "--runs", f"{d.out}/runs", "--tiling", str(d.tiling),
              "--slide-csv", str(d.slide_csv)]  # fmt: skip
    if t.threads:
        common += ["--threads", str(t.threads)]
    train = ["train", *common, "--variants", variant,
             "--folds", *map(str, e.folds), "--epochs", str(t.epochs),
             "--neg-cap", str(t.neg_cap), "--cap-seed", str(t.cap_seed)]  # fmt: skip
    if t.smoke_tiles_per_slide is not None:
        train += ["--smoke-tiles-per-slide", str(t.smoke_tiles_per_slide)]
    # Step-8 rules: exactly run_cv.sh's call; other rules: those of the cache
    rules = ["--label-rules-from-cache"] if label_rules_diff(config) else []
    train += rules
    if not e.report:
        return train, None
    report = ["report", *common, "--annotations", str(d.annotations),
              "--n-boot", str(e.n_boot), "--out", str(d.out), *rules]  # fmt: skip
    if e.a_runs:
        report += ["--a-runs", str(e.a_runs)]
    if e.other_runs:
        report += ["--other-runs", *map(str, e.other_runs)]
    return train, report


def feature_caches(config: DictConfig) -> tuple[str, str]:
    """Build or reuse the caches (same keys as run_cv.sh, so existing ones are reused)."""
    from ml.data.huncrc_features import build_features
    from ml.data.huncrc_labels import LabelConfig

    bfc = _script("build_feature_cache")
    d, rules = config.data, label_config(config)
    table = bfc.slide_table(Path(d.tiling), Path(d.slide_csv))
    ann = Path(d.annotations) if rules.needs_annotations else None
    dim = int(d.embedding_dim)
    model = str(d.embedding_model)
    train = build_features(Path(d.embeddings), table, ["train"], rules,
                           Path(d.cache_root), dim=dim, annotations=ann,
                           model=model)  # fmt: skip
    prelim = build_features(Path(d.embeddings), table, ["test_preliminary"],
                            LabelConfig(), Path(d.cache_root), dim=dim,
                            mode="all", model=model)  # fmt: skip
    return str(train), str(prelim)


def epoch_hook(
    logger: OptionalMLFlowLogger,
) -> Callable[[str, int, dict[str, Any]], None]:
    def on_epoch(variant: str, k: int, rec: dict[str, Any]) -> None:
        keys = ["train_loss", "val_auroc", "val_auprc", "seconds"]
        keys += sorted(key for key in rec if key.startswith("monitor_"))  # torchmetrics
        for key in keys:
            logger.log_series(f"fold{k}_epoch_{key}", [(int(rec["epoch"]), rec[key])])

    return on_epoch


def log_final(config: DictConfig, logger: OptionalMLFlowLogger, variant: str) -> None:
    """After the report: the log_to_mlflow.py plan, into this run."""
    if label_rules_diff(config):
        print("[mlflow] final metrics not logged: label rules differ from Step 8, "
              "and log_to_mlflow.py describes the Step-8 sets only", flush=True)  # fmt: skip
        return
    ltm = _script("log_to_mlflow")
    out = Path(config.data.out)
    try:
        report = ltm._read_json(out / "report.json")
        plan = ltm.build_plan(variant, report, out / "runs" / variant, out / "report.json",
                              (out / "table.txt").read_text())  # fmt: skip
    except ltm.RefuseError as e:  # e.g. a partial or smoke run: disk only
        print(f"[mlflow] final metrics not logged: {e}", flush=True)
        return
    tags = {k: v for k, v in plan.tags.items() if k != "retrospective"}
    logger.set_tags({**tags, "retrospective": "false", "entry_point": "ml.huncrc_cv"})
    logger.log_hyperparams(plan.params)
    logger.log_metrics(plan.metrics, step=0)
    for name, points in plan.series.items():
        logger.log_series(name, points)
    for name, text in plan.artifacts.items():
        logger.log_text_artifact(text, name)


# ------------------------------------------------------------------ main


@hydra.main(config_path="../configs", config_name="huncrc_tile", version_base=None)
@checked
@autolog  # type: ignore[arg-type]  # our logger is a subclass of the house one
def main(config: DictConfig, logger: OptionalMLFlowLogger) -> None:
    from ml._mlflow_compat import apply_mlflow_compat_patch

    apply_mlflow_compat_patch()
    variant = check_config(config)
    wcv = _script("within_slide_cv")
    warning = label_rules_warning(config)
    if warning:
        print(warning, file=sys.stderr, flush=True)
        diff = label_rules_diff(config)
        logger.set_tags({"label_rules": "differ from Step 8",
                         "label_rules_diff": ", ".join(diff)[:250]})  # fmt: skip
    if config.dry_run:
        train, report = build_calls(config, variant, "<train cache>", "<prelim cache>")
        print("dry run: caches would be built or reused under", config.data.cache_root)
        print("within_slide_cv.py", " ".join(train))
        if report:
            print("within_slide_cv.py", " ".join(report))
        have = [v for v in AUTH_VARS if os.environ.get(v)]
        print("mlflow: not contacted in a dry run; credential variables set:",
              ", ".join(have) or "none", "(mlflow.login=true would prompt)")  # fmt: skip
        return
    train, report = build_calls(config, variant, *feature_caches(config))
    wcv.main(train, on_epoch=epoch_hook(logger))
    if report is not None:
        wcv.main(report)
        log_final(config, logger, variant)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
