"""IBD-34 Step 2: pick one fixed threshold per model on its val fold (CPU-only).

Consumes the per-run val predictions parquets that the val-fold job (Step 1)
logged under a single MLflow run, then for each model sweeps thresholds and
selects the argmax-F1 cut via :func:`ml.inference.f1_scan`. Emits a per-model
table: training run_id, the chosen threshold, and its F1 / precision / recall.
No GPU and no model reload -- the forward pass already happened in Step 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from datasets import Dataset
from mlflow.store.artifact import artifact_repository_registry as _artifact_registry
from omegaconf import DictConfig

from ml.inference import f1_scan


if TYPE_CHECKING:
    from rationai.mlkit.lightning.loggers import MLFlowLogger


def _prediction_parquet_uris(valfold_run_id: str) -> list[tuple[str, str]]:
    """Return [(training_run_id, predictions parquet uri)] under ``valfold/``."""
    repo = _artifact_registry.get_artifact_repository(f"runs:/{valfold_run_id}")
    entries = repo._list_run_artifacts("valfold")
    uris: list[tuple[str, str]] = []
    for entry in entries:
        if not entry.is_dir:
            continue
        train_id = Path(entry.path).name
        uris.append(
            (
                train_id,
                f"runs:/{valfold_run_id}/valfold/{train_id}/val_predictions.parquet",
            )
        )
    return sorted(uris)


def _score_parquet(path: str | Path, num_thresholds: int) -> dict[str, object]:
    result = f1_scan(path, num_thresholds=num_thresholds)
    return {
        "threshold": float(result["threshold"]),
        "precision": float(result["precision"]),
        "recall": float(result["recall"]),
        "f1": float(result["f1"]),
    }


def threshold_main(config: DictConfig, logger: MLFlowLogger) -> None:
    valfold_run_id = str(config.valthreshold.valfold_run_id)
    num_thresholds = int(config.valthreshold.num_thresholds)
    print(
        f"Picking a val-fold threshold per model (sweep of {num_thresholds} cuts) "
        f"from valfold run {valfold_run_id}.",
        flush=True,
    )

    pairs = _prediction_parquet_uris(valfold_run_id)
    if not pairs:
        print(f"No val-fold predictions under run {valfold_run_id}.", flush=True)
        return
    print(f"Found {len(pairs)} model(s) to threshold.", flush=True)

    rows: list[dict[str, object]] = []
    for train_id, uri in pairs:
        local = mlflow.artifacts.download_artifacts(uri)
        row: dict[str, object] = {
            "run_id": train_id,
            **_score_parquet(local, num_thresholds),
        }
        rows.append(row)
        print(
            f"[{train_id}] threshold={row['threshold']:.4f} "
            f"f1={row['f1']:.4f} "
            f"(precision={row['precision']:.4f}, recall={row['recall']:.4f})",
            flush=True,
        )
        logger.log_metrics(
            {
                f"{train_id}/threshold": float(row["threshold"]),
                f"{train_id}/f1": float(row["f1"]),
            }
        )

    out = Path("valthreshold/thresholds.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(out)
    logger.log_artifact(str(out), artifact_path="valthreshold")

    run = logger.experiment.get_run(logger.run_id)
    uri = (
        f"mlflow-artifacts:/{run.info.experiment_id}/{logger.run_id}"
        "/artifacts/valthreshold/thresholds.parquet"
    )
    print(f"Threshold table stored in MLflow: {uri} ({len(rows)} rows)", flush=True)
