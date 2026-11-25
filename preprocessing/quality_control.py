r"""Ask the QC RayService to mask every slide, then stage the masks for MLflow.

    uv run python -m preprocessing.quality_control +dataset=processed
    (submitted as a job by scripts/preprocessing/quality_control.py)

HUNCRC LEDGER as at 2026-10-07 12:56 -- job launched 2026-10-01T16:13:14Z, still Running
after 5 d 20 h.  Asked for 200 slides:  186 done + 13 dead + 1 in flight = 200.

    dead (error line, no csv)          011 036 039 042 045 059 064
                                       086 091 125 155 162 174
    in flight (neither csv nor error)  194
    timed out but recovered anyway     002 006 019 024 030 053 088 093 094 149 160 176 182

Recompute rather than trusting these lists:

    comm -23 /tmp/err.txt /tmp/done.txt                          # dead: retry
    comm -23 /tmp/all.txt <(sort -u /tmp/done.txt /tmp/err.txt)  # in flight: wait

where done = *.csv stems in output_dir, err = slide ids in qc_errors.log, all = *.mrxs
stems under the slides dir.

WHY MISSING. One slide costs ~3.6 h of service time (186 csvs over 132.4 h = one every
42.7 min, x max_concurrent 5 => 3.6 h each), so 200/5 x 3.6 h ~= 6 days -- the 5 days are
normal, not a stall. request_timeout is 5 h (configs/preprocessing/qc.yaml:4), so timing
out is business as usual. qc_errors.log has two kinds of line and the text after the colon
is the whole diagnosis:
  * 25 lines end with NOTHING. `str(httpx.ReadTimeout())` is `''`, so this is the client
    quitting at the 5 h ceiling; it says nothing about the slide. The service keeps
    working after the client leaves, which is how 13 of those 25 later got a csv.
  * slide 042 reads `RetryError[...HTTPStatusError]` -- the service answered 500 three
    times (@retry only fires on 500). A real service error; may fail again on retry.
Nothing is ever retried: check_slides sends each slide once and @retry ignores timeouts,
so a slide whose service run finishes 2 min late is abandoned anyway.

A GREEN EXIT PROVES NOTHING -- append_error_log is the only record, and qc_main returns
normally with 13 slides missing. Worth making it fail instead: refuse to upload a partial
mask tree (see preprocessing/annotation_masks.py) and raise when qc_errors.log is
non-empty. tiling.py needs what is absent today -- qc_metrics.csv (:216), masks under
subdirs organize_masks creates (:64), and a csv row per slide, since :31 does df.loc[stem]
and raises KeyError on one of the 13.

The 13 do not need a re-run of all 200 (~6 days). Retry them directly, ~9 h at
max_concurrent 5, once the service is free:

    for s in 011 036 039 042 045 059 064 086 091 125 155 162 174; do
        uv run python scripts/qc_one_slide.py --slide $s.mrxs \
            --out /mnt/projects/huncrc_dysplasia/quality_control --timeout 18000
    done

On hunCRC the job's own tail has never run (no subdirs, no qc_metrics.csv), so masks are
still flat `<Prefix>_<slide>.tiff` and anything added later lands flat beside them.
"""

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import hydra
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.client import AsyncClient
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.types import SlideCheckConfig
from tqdm.asyncio import tqdm


class QCParameters(TypedDict):
    mask_level: int
    sample_level: int
    check_residual: bool
    check_folding: bool
    check_focus: bool
    wb_correction: bool


def get_qc_masks(qc_parameters: QCParameters) -> Generator[tuple[str, str], None, None]:
    if qc_parameters["check_focus"]:
        yield ("Piqe_focus_score_piqe_median", "blur_per_tile")
        yield ("Piqe_piqe_median_activity_mask", "blur_per_pixel")

    if qc_parameters["check_residual"]:
        yield ("ResidualArtifactsAndCoverage_cov_percent_heatmap", "artifacts_per_tile")
        yield ("ResidualArtifactsAndCoverage_coverage_mask", "artifacts_per_pixel")

    if qc_parameters["check_folding"]:
        yield ("FoldingFunction_folding_test", "folds_per_pixel")


def organize_masks(output_path: Path, subdir: str, mask_prefix: str) -> None:
    prefix_dir = output_path / subdir
    prefix_dir.mkdir(parents=True, exist_ok=True)

    for file in list(output_path.glob(f"{mask_prefix}_*.tiff")):
        slide_name = file.name.replace(f"{mask_prefix}_", "")
        destination = prefix_dir / slide_name
        file.rename(destination)


def append_error_log(output_path: Path, message: str) -> None:
    with open(output_path / "qc_errors.log", "a") as log_file:
        log_file.write(message)


async def qc_main(
    output_path: Path,
    slides: list[str],
    request_timeout: int,
    max_concurrent: int,
    qc_parameters: QCParameters,
) -> None:
    async with AsyncClient() as client:
        async for result in tqdm(
            client.qc.check_slides(
                slides,
                output_path,
                config=SlideCheckConfig(**qc_parameters),
                timeout=request_timeout,
                max_concurrent=max_concurrent,
            ),
            total=len(slides),
        ):
            if not result.success:
                await asyncio.to_thread(
                    append_error_log,
                    output_path,
                    f"Failed to process {result.wsi_path}: {result.error}\n",
                )

        for prefix, artifact_name in get_qc_masks(qc_parameters):
            organize_masks(output_path, artifact_name, prefix)

        csvs = list(output_path.glob("*.csv"))
        pd.concat([pd.read_csv(f) for f in csvs]).to_csv(
            output_path / "qc_metrics.csv", index=False
        )

        for f in csvs:
            f.unlink()


@with_cli_args(["+preprocessing=qc"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset = pd.read_csv(
        download_artifacts(artifact_uri=config.dataset.mlflow_uris.dataset)
    )

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        qc_main(
            output_path=output_path,
            slides=dataset["slide_path"].to_list(),
            request_timeout=config.request_timeout,
            max_concurrent=config.max_concurrent,
            qc_parameters=config.qc_parameters,
        )
    )

    logger.log_artifacts(str(output_path), "qc_output")


if __name__ == "__main__":
    main()
