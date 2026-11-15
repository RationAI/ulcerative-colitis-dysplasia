"""IBD-34 Step 1: run each trained model over its own validation fold.

Driven by a checked-in manifest (``ml/manifests/level1_high.yaml``): one row
per training run, each carrying the run metadata (``fold``, ``epi``,
``negative_slides``) and the MLflow checkpoint URI. Nothing is discovered from
the experiment or parsed from resolved configs. For every row this builds the
labeled validation fold from the row's filter settings, forwards the head once
over it, and logs a per-run predictions parquet (with ``hg_label``) to the
current MLflow run. No threshold selection, no F1, no heatmaps -- just the
val-fold predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml.data.datasets import Embeddings
from ml.data.datasets.labels import LabelMode
from ml.inference import collect, write_parquet


if TYPE_CHECKING:
    from rationai.mlkit.lightning.loggers import MLFlowLogger


def _resolve_checkpoint(checkpoint_uri: str | None) -> str | None:
    if checkpoint_uri and str(checkpoint_uri).startswith("mlflow-artifacts:/"):
        return mlflow.artifacts.download_artifacts(str(checkpoint_uri))
    return checkpoint_uri


def _load_model(
    model_cfg: DictConfig, checkpoint_uri: str, device: torch.device
) -> torch.nn.Module:
    model = hydra.utils.instantiate(model_cfg)
    ckpt = _resolve_checkpoint(checkpoint_uri)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return model


def _loader_len(loader: DataLoader) -> int | None:
    try:
        return len(loader)
    except (TypeError, NotImplementedError):
        return None


def valfold_main(config: DictConfig, logger: MLFlowLogger) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Val-fold predictions on device: {device}", flush=True)

    manifest_path = config.valfold.manifest
    rows = OmegaConf.load(manifest_path)
    if not isinstance(rows, (list, tuple)):
        rows = rows.runs
    print(
        f"Scoring {len(rows)} run(s) from {manifest_path} over their val folds.",
        flush=True,
    )

    mode = LabelMode(config.label_mode)
    batch_size = int(config.datamodule.batch_size)
    num_workers = int(config.datamodule.num_workers)
    base_thresholds = dict(config.dataset.thresholds)
    base_min_thresholds = dict(config.dataset.min_thresholds)

    run = logger.experiment.get_run(logger.run_id)
    for row in rows:
        run_id = str(row.run_id)
        fold = int(row.fold)
        epi = float(row.get("epi", 0.0))
        negative_slides = bool(row.negative_slides)

        model = _load_model(config.model, str(row.checkpoint_uri), device)

        # Match the training filter distribution for this run's val fold.
        min_thresholds = {**base_min_thresholds, "epithelium": epi}
        print(
            f"[{run_id}] loading val-fold embeddings "
            f"(fold {fold}, epi {epi}, neg {'on' if negative_slides else 'off'})...",
            flush=True,
        )
        dataset = Embeddings(
            uris=[config.dataset.mlflow_uris.embeddings.train],
            mode=mode,
            thresholds=base_thresholds,
            min_thresholds=min_thresholds,
            negative_slides=negative_slides,
            val_fold=fold,
            is_val=True,
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        collected = collect(
            tqdm(loader, desc=f"run {run_id} (fold {fold})", total=_loader_len(loader)),
            model,
            device,
            include_labels=True,
            mode=mode,
        )

        out = write_parquet(collected, f"valfold/{run_id}/val_predictions.parquet")
        logger.log_artifact(str(out), artifact_path=f"valfold/{run_id}")
        uri = (
            f"mlflow-artifacts:/{run.info.experiment_id}/{logger.run_id}"
            f"/artifacts/valfold/{run_id}/val_predictions.parquet"
        )
        print(
            f"[{run_id}] fold {fold} epi {epi} "
            f"neg {'on' if negative_slides else 'off'} "
            f"-> {len(collected)} tiles -> {uri}",
            flush=True,
        )
