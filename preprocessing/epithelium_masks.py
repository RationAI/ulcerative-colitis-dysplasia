import asyncio
from pathlib import Path

import hydra
import mlflow.artifacts
import pandas as pd
from omegaconf import DictConfig
from rationai.client import AsyncClient
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm


async def segment_epithel(
    slides: list[str], tissue_masks_dir: Path, output_dir: Path, max_concurrent: int
) -> None:
    async with AsyncClient(
        timeout=3000,
    ) as client:
        pending: set[asyncio.Task[str]] = set()

        with tqdm(total=len(slides), desc="Processing slides") as pbar:
            for path in slides:
                if len(pending) >= max_concurrent:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        try:
                            t.result()
                        except Exception as e:
                            print(f"Slide processing failed: {e}")
                    pbar.update(len(done))

                tissue_mask_path = (
                    tissue_masks_dir / Path(path).with_suffix(".tiff").name
                )
                output_path = output_dir / Path(path).with_suffix(".tiff").name

                task = client.slide.heatmap(
                    "episeg-1",
                    slide_path=path,
                    tissue_mask_path=tissue_mask_path.as_posix(),
                    output_path=output_path.as_posix(),
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
    asyncio.run(
        segment_epithel(
            slides=dataset["slide_path"].tolist(),
            tissue_masks_dir=tissue_masks_path,
            output_dir=output_dir,
            max_concurrent=config.max_concurrent,
        )
    )
    logger.log_artifacts(str(output_dir), config.mlflow_artifact_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
