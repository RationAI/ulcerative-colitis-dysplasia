import asyncio
from pathlib import Path

import hydra
import httpx
import mlflow.artifacts
import pandas as pd
from omegaconf import DictConfig
from rationai.client import AsyncClient
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm


async def segment_epithel(
    slides: list[str],
    tissue_masks_dir: Path,
    output_dir: Path,
    max_concurrent: int,
    model: str,
    svc: str,
    request_timeout: int,
) -> None:
    #: models_base_url is REQUIRED, not a nicety: AsyncClient defaults to IBD's
    #: rayservice-model-serve-svc, which on the public cluster has zero live replicas
    #: for hunCRC and would time out every request. request_timeout bounds the client
    #: connection; the SDK's per-request heatmap default (1000 s) is too short for
    #: hunCRC -- the first request after a reroll cold-starts the GPU and rebuilds the
    #: TensorRT engine in ~1544 s measured -- so it is passed explicitly as well.
    async with AsyncClient(
        models_base_url=svc,
        timeout=request_timeout,
    ) as client:
        pending: set[asyncio.Task[str]] = set()

        with tqdm(total=len(slides), desc="Processing slides") as pbar:
            for path in slides:
                if len(pending) >= max_concurrent:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        error = t.exception()
                        if error is not None:
                            print(f"Slide processing failed: {error}")
                    pbar.update(len(done))

                tissue_mask_path = (
                    tissue_masks_dir / Path(path).with_suffix(".tiff").name
                )
                output_path = output_dir / Path(path).with_suffix(".tiff").name

                task = client.slide.heatmap(
                    model,
                    slide_path=path,
                    tissue_mask_path=tissue_mask_path.as_posix(),
                    output_path=output_path.as_posix(),
                    timeout=request_timeout,
                )
                pending.add(asyncio.create_task(task))

            if pending:
                results = await asyncio.gather(*pending, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        print(f"Slide processing failed: {r}")
            pbar.update(len(pending))


@with_cli_args(["+preprocessing=epithelium_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=config.dataset.mlflow_uris.dataset
        )
    )
    dataset = pd.read_csv(dataset_path)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    #: Captured BEFORE the only_missing filter, because that is the completeness bar for
    #: the upload at the end: on a resume, dataset holds only the missing slides, but
    #: output_dir already holds the good masks from the earlier run.
    total_slides = len(dataset)
    if config.only_missing:
        existing_masks = {p.name for p in output_dir.glob("*.tiff")}
        dataset = dataset[
            ~dataset["slide_path"]
            .apply(lambda p: Path(p).with_suffix(".tiff").name)
            .isin(existing_masks)
        ]

    tissue_masks_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=config.dataset.mlflow_uris.tissue,
            dst_path=config.project_path,
        )
    )
    #: Pre-flight the service once, cheaply: if it is down, failing after downloading
    #: tissue masks but before the first request would look identical to a hung run.
    try:
        httpx.get(f"{config.svc}/openapi.json", timeout=180, follow_redirects=True)
    except Exception as e:
        raise SystemExit(
            f"model service {config.svc} did not answer /openapi.json: "
            f"{type(e).__name__}: {e}. A rerolled deployment can take minutes to "
            "spawn its first replica -- wait and re-run (only_missing resumes), do "
            "not start the job against a dead service."
        ) from e

    asyncio.run(
        segment_epithel(
            slides=dataset["slide_path"].tolist(),
            tissue_masks_dir=tissue_masks_path,
            output_dir=output_dir,
            max_concurrent=config.max_concurrent,
            model=config.model,
            svc=str(config.svc),
            request_timeout=int(config.request_timeout),
        )
    )
    #: Fail closed: per-slide errors only print and move on (see the task handlers
    #: above), so a partially-segmented cohort would otherwise be uploaded as
    #: `epithelium_masks` and silently consumed downstream. Count before uploading and
    #: refuse a partial tree (same rule as preprocessing/annotation_masks.py).
    produced = len(list(output_dir.glob("*.tiff")))
    if produced < total_slides:
        raise SystemExit(
            f"refusing to log a partial tree: {produced}/{total_slides} masks in "
            f"{output_dir}. Re-run to resume (only_missing) and finish the rest."
        )
    logger.log_artifacts(str(output_dir), config.mlflow_artifact_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
