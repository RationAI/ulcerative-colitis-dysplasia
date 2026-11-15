"""IBD-34 Step 3b: final test_preliminary F1, per model, aggregated per config.

Consumes the two prior steps' outputs:
  * Step 3a test predictions: ``artifacts/test_preliminary/<train_run_id>/test_predictions.parquet``
    (each row has ``prob_high`` and the ground-truth ``hg_label``);
  * Step 2 thresholds: ``artifacts/valthreshold/thresholds.parquet`` (per train
    run id, the val-fold argmax-F1 threshold).

For every model it computes test_preliminary F1/P/R at that model's own
threshold, then aggregates the 5 folds of each config (fold/epi/negative-slide
setting from the manifest) into a mean +/- std. Logs the per-model table and the
per-config aggregate under one MLflow run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pandas as pd
from datasets import Dataset
from mlflow.store.artifact import artifact_repository_registry as _artifact_registry
from omegaconf import DictConfig, OmegaConf

from ml.inference import f1_at


if TYPE_CHECKING:
    from rationai.mlkit.lightning.loggers import MLFlowLogger


def _prediction_parquet_uris(run_id: str, phase: str) -> list[tuple[str, str]]:
    """Return [(train_run_id, predictions parquet uri)] under ``<phase>/``."""
    repo = _artifact_registry.get_artifact_repository(f"runs:/{run_id}")
    entries = repo._list_run_artifacts(phase)
    uris: list[tuple[str, str]] = []
    for entry in entries:
        if not entry.is_dir:
            continue
        train_id = Path(entry.path).name
        uris.append(
            (train_id, f"runs:/{run_id}/{phase}/{train_id}/test_predictions.parquet")
        )
    return sorted(uris)


def _config_key(row) -> str:
    epi = float(row.get("epi", 0.0))
    neg = bool(row.negative_slides)
    return f"epi{epi:g}_neg{'on' if neg else 'off'}"


def _aggregate(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    for row in results:
        grouped.setdefault(str(row["config"]), []).append(float(row["f1"]))

    rows: list[dict[str, object]] = []
    for config in sorted(grouped):
        f1s = grouped[config]
        mean = sum(f1s) / len(f1s)
        std = (
            ((sum((v - mean) ** 2 for v in f1s) / len(f1s)) ** 0.5)
            if len(f1s) > 1
            else 0.0
        )
        rows.append(
            {
                "config": config,
                "n_folds": len(f1s),
                "test_f1_mean": mean,
                "test_f1_std": std,
            }
        )
    return rows


def testf1_main(config: DictConfig, logger: MLFlowLogger) -> None:
    testprelim_run_id = str(config.testf1.testprelim_run_id)
    thresholds_run_id = str(config.testf1.thresholds_run_id)
    print(
        f"Final test_preliminary F1 from testprelim run {testprelim_run_id} "
        f"using thresholds from run {thresholds_run_id}.",
        flush=True,
    )

    # Step 2 thresholds: run_id -> threshold.
    thr_local = mlflow.artifacts.download_artifacts(
        f"runs:/{thresholds_run_id}/valthreshold/thresholds.parquet"
    )
    thr_df = pd.read_parquet(thr_local)
    thresholds = {str(r["run_id"]): float(r["threshold"]) for _, r in thr_df.iterrows()}

    # Manifest: run_id -> fold/epi/negative_slides (for config grouping).
    manifest = OmegaConf.load(config.testf1.manifest)
    mrows = manifest.runs if "runs" in manifest else manifest
    meta = {str(r.run_id): r for r in mrows}

    pairs = _prediction_parquet_uris(testprelim_run_id, "test_preliminary")
    if not pairs:
        print(
            f"No test_preliminary predictions under run {testprelim_run_id}.",
            flush=True,
        )
        return
    print(f"Found {len(pairs)} model(s) to score.", flush=True)

    per_run: list[dict[str, object]] = []
    for train_id, uri in pairs:
        if train_id not in thresholds:
            print(f"[{train_id}] no threshold found, skipping", flush=True)
            continue
        threshold = thresholds[train_id]
        local = mlflow.artifacts.download_artifacts(uri)
        stats = f1_at(local, threshold)
        meta_row = meta.get(train_id)
        fold = int(meta_row.fold) if meta_row is not None else -1
        row: dict[str, object] = {
            "run_id": train_id,
            "config": _config_key(meta_row) if meta_row is not None else "?",
            "fold": fold,
            "threshold": threshold,
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1": float(stats["f1"]),
        }
        per_run.append(row)
        print(
            f"[{train_id}] config={row['config']} fold {fold} "
            f"threshold={threshold:.4f} test_f1={row['f1']:.4f} "
            f"(p={row['precision']:.4f}, r={row['recall']:.4f})",
            flush=True,
        )
        logger.log_metrics({f"{train_id}/test_f1": float(row["f1"])})

    aggregate = _aggregate(per_run)
    out_dir = Path("testf1")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = out_dir / "test_f1_per_run.parquet"
    agg_path = out_dir / "test_f1_aggregate.parquet"
    Dataset.from_list(per_run).to_parquet(per_run_path)
    Dataset.from_list(aggregate).to_parquet(agg_path)
    logger.log_artifact(str(per_run_path), artifact_path="testf1")
    logger.log_artifact(str(agg_path), artifact_path="testf1")

    run = logger.experiment.get_run(logger.run_id)
    print(
        f"Per-run + aggregate F1 stored under mlflow-artifacts:/"
        f"{run.info.experiment_id}/{logger.run_id}/artifacts/testf1",
        flush=True,
    )
    print("\nPer-config test_preliminary F1 (mean +/- std over folds):", flush=True)
    for row in aggregate:
        print(
            f"{row['config']!s:18} f1={row['test_f1_mean']:.4f} +/- "
            f"{row['test_f1_std']:.4f} (n={row['n_folds']})",
            flush=True,
        )
