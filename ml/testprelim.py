"""IBD-34 Step 3a: run each trained model on the held-out test_preliminary set.

Manifest-driven like Step 1: every row of the checked-in manifest supplies a
training run's checkpoint URI, and this scores it over the (shared) test
preliminary tiles. Each parquet carries ``prob_high`` and ``hg_label`` (the
test set is labeled); computing F1 is Step 3b, which reads these parquets and
applies each model's Step-2 threshold. Because all models share the same
test_preliminary source, the dataset and DataLoader are built once and reused
across runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ml.data.datasets import Embeddings
from ml.data.datasets.labels import LabelMode
from ml.inference import collect, write_parquet
from ml.valfold import _load_model, _loader_len


if TYPE_CHECKING:
    from rationai.mlkit.lightning.loggers import MLFlowLogger


def testprelim_main(config: DictConfig, logger: MLFlowLogger) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"test_preliminary predictions on device: {device}", flush=True)

    manifest_path = config.testprelim.manifest
    rows = OmegaConf.load(manifest_path)
    if not isinstance(rows, (list, tuple)):
        rows = rows.runs
    print(
        f"Scoring {len(rows)} model(s) from {manifest_path} over test_preliminary.",
        flush=True,
    )

    mode = LabelMode(config.label_mode)
    batch_size = int(config.datamodule.batch_size)
    num_workers = int(config.datamodule.num_workers)

    # Labeled source so each test parquet carries ``hg_label`` alongside
    # ``prob_high`` (Step 3b needs the ground truth to compute F1). All models
    # score the same fixed test set: keep every tile, no mask filtering.
    dataset = Embeddings(
        uris=[config.dataset.mlflow_uris.embeddings.test_preliminary],
        mode=mode,
        thresholds=dict(config.testprelim.thresholds),
        min_thresholds=dict(config.testprelim.min_thresholds),
        negative_slides=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    print(f"test_preliminary tiles: {len(loader)} batches.", flush=True)

    run = logger.experiment.get_run(logger.run_id)
    for row in rows:
        run_id = str(row.run_id)
        model = _load_model(config.model, str(row.checkpoint_uri), device)

        collected = collect(
            tqdm(
                loader,
                desc=f"test_prelim {run_id}",
                total=_loader_len(loader),
            ),
            model,
            device,
            include_labels=True,
            mode=mode,
        )

        out = write_parquet(
            collected, f"test_preliminary/{run_id}/test_predictions.parquet"
        )
        logger.log_artifact(str(out), artifact_path=f"test_preliminary/{run_id}")
        uri = (
            f"mlflow-artifacts:/{run.info.experiment_id}/{logger.run_id}"
            f"/artifacts/test_preliminary/{run_id}/test_predictions.parquet"
        )
        print(
            f"[{run_id}] -> {len(collected)} tiles -> {uri}",
            flush=True,
        )
