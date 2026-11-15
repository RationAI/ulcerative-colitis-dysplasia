from pathlib import Path
from random import randint

import hydra
import mlflow
import torch
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from ml._mlflow_compat import apply_mlflow_compat_patch
from ml.data import DataModule
from ml.heatmap import save_heatmaps
from ml.heatmap_report import heatmap_report_main
from ml.inference import collect, f1_scan, write_parquet
from ml.predict_output import _class_names
from ml.testf1 import testf1_main
from ml.testprelim import testprelim_main
from ml.valfold import valfold_main
from ml.valthreshold import threshold_main


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


def _resolve_checkpoint(config: DictConfig) -> str | None:
    ckpt = config.checkpoint
    if ckpt is not None and str(ckpt).startswith("mlflow-artifacts:/"):
        return mlflow.artifacts.download_artifacts(str(ckpt))
    return ckpt


def _predict(config: DictConfig, logger: MLFlowLogger, model: torch.nn.Module) -> None:
    """Run the head once over a chosen tile source and apply post-processors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = config.predict

    ckpt = _resolve_checkpoint(config)
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["state_dict"])
        model.to(device).eval()

    if p.source is not None:
        dataset = hydra.utils.instantiate(p.source)
    else:
        data = hydra.utils.instantiate(
            config.datamodule,
            _recursive_=False,  # to avoid instantiating all the datasets
            _target_=DataModule,
        )
        dataset = data.predict

    loader = DataLoader(
        dataset,
        batch_size=config.datamodule.batch_size,
        num_workers=config.datamodule.num_workers,
    )
    rows = collect(
        loader,
        model,
        device,
        include_labels=bool(p.get("labels", False)),
        mode=config.label_mode,
    )
    write_parquet(rows, p.predict_output)

    run = logger.experiment.get_run(logger.run_id)
    logger.log_artifact(p.predict_output, artifact_path="predictions")
    uri = (
        f"mlflow-artifacts:/{run.info.experiment_id}/{logger.run_id}"
        f"/artifacts/predictions/{Path(p.predict_output).name}"
    )
    print(f"Predictions stored in MLflow: {uri}", flush=True)

    postprocess: list[str] = list(p.get("postprocess") or [])
    if "f1" in postprocess:
        result = f1_scan(p.predict_output, num_thresholds=int(p.f1.num_thresholds))
        logger.log_metrics(
            {f"f1_scan/{name}": float(value) for name, value in result.items()}
        )
        print(
            f"argmax-F1 threshold={result['threshold']:.4f} "
            f"f1={result['f1']:.4f} "
            f"(precision={result['precision']:.4f}, recall={result['recall']:.4f})",
            flush=True,
        )
    if "heatmap" in postprocess:
        heatmap_dir = "heatmaps"
        saved = save_heatmaps(
            p.predict_output,
            dataset.slides,
            _class_names(config.label_mode),
            heatmap_dir,
        )
        if saved:
            logger.log_artifact(heatmap_dir, artifact_path="heatmaps")
            uri = (
                f"mlflow-artifacts:/{run.info.experiment_id}/{logger.run_id}"
                "/artifacts/heatmaps"
            )
            print(
                f"Heatmaps stored in MLflow: {uri} ({len(saved)} files)",
                flush=True,
            )


@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    apply_mlflow_compat_patch()

    seed_everything(config.seed, workers=True)

    model = hydra.utils.instantiate(config.model)

    if config.mode in ("fit", "validate", "test"):
        data = hydra.utils.instantiate(
            config.datamodule,
            _recursive_=False,  # to avoid instantiating all the datasets
            _target_=DataModule,
        )
        trainer = hydra.utils.instantiate(
            config.trainer, _target_=Trainer, logger=logger
        )
        getattr(trainer, config.mode)(
            model, datamodule=data, ckpt_path=_resolve_checkpoint(config)
        )
    elif config.mode == "predict":
        _predict(config, logger, model)
    elif config.mode == "valfold":
        valfold_main(config, logger)
    elif config.mode == "valthreshold":
        threshold_main(config, logger)
    elif config.mode == "testprelim":
        testprelim_main(config, logger)
    elif config.mode == "testf1":
        testf1_main(config, logger)
    elif config.mode == "heatmap":
        heatmap_report_main(config, logger)
    mlflow.end_run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
