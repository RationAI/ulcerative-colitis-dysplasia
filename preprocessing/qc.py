import asyncio
import tempfile
from pathlib import Path

import hydra
import rationai
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm.asyncio import tqdm


async def qc_main(output_path: str, report_path: str, slides: list[str]) -> None:
    async with rationai.AsyncClient() as client:
        async for result in tqdm(
            client.qc.check_slides(slides, output_path),
            total=len(slides),
        ):
            if not result.success:
                with open(Path(output_path) / "qc_errors.log", "a") as log_file:
                    log_file.write(
                        f"Failed to process {result.wsi_path}: {result.error}\n"
                    )

        await client.qc.generate_report(
            backgrounds=slides,
            mask_dir=output_path,
            save_location=report_path,
            compute_metrics=True,
        )


@hydra.main(config_path="../configs", config_name="preprocessing/qc", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    slides: list[str] = ...  # Load slide paths

    with tempfile.TemporaryDirectory(
        prefix="qc_masks_report_", dir=config.project_dir
    ) as tmp_dir:
        asyncio.run(
            qc_main(
                output_path=output_path.absolute().as_posix(),
                report_path=tmp_dir,
                slides=slides,
            )
        )
        logger.log_artifacts(local_dir=tmp_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
