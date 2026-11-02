#: Provenance cut-down of the QC step.
#:
#: The original preprocessing/quality_control.py is left UNTOUCHED. This copy
#: reuses its qc_main() (one source of truth for the logic) on a 2-slide
#: subset and wraps it in a provenance-logged MLflow run:
#:
#:   1. verify_manifest — the registered cohort still matches the pinned run
#:   2. qc_main         — the original logic, 2 slides -> the shared QC service
#:   3. gate            — FAIL LOUDLY if qc_metrics.csv has < 2 rows (the
#:                        original exits green with missing slides; a provenance
#:                        run for partial QC would be a lie)
#:   4. log_provenance  — env + hardware, config params, qc_metrics output +
#:                        the pinned input run (runs:/<create_dataset run id>)
#:
#: SLOW BY NATURE: the QC service costs ~3.6 h of service time per slide, so
#: even the 2-slide cut-down takes hours. Everything else in the provenance
#: chain is minutes.
#:
#: The masks land in a PROV-dedicated staging dir (never the production
#: quality_control/ tree) and are uploaded as the `qc_output` artifact that
#: the tiling cut-down reads.
#:
#: Run as a kube-job (submitter: provenance/submitters/quality_control.py;
#: needs JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.quality_control
import asyncio
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.artifacts import download_artifacts
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest

#: Reuse the original logic — the original module stays byte-identical.
from preprocessing.quality_control import qc_main

#: Where the cut-down test runs live, isolated from the production runs.
EXPERIMENT = "provenance_huncrc_test"
#: Pinned create_dataset cut-down run (provenance/create_dataset.py) — its
#: dataset.csv artifact IS the 2-slide input for this whole test chain.
CREATE_DATASET_RUN_ID = "bc5160ba96064230b375cb15fdf78320"
#: Pinned registration run of the huncrc cohort (verify gate).
DATASET_RUN_ID = "4ef2fb98adb8432b9ee224fc5bdd5231"

#: Same values as configs/preprocessing/qc.yaml — hardcoded because this script
#: intentionally does not go through the hydra config tree.
QC_PARAMETERS = {
    "mask_level": 3,
    "sample_level": 1,
    "check_residual": True,
    "check_folding": False,
    "check_focus": True,
    "wb_correction": True,
}
REQUEST_TIMEOUT = 18000
MAX_CONCURRENT = 5

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
#: Prov-dedicated staging dir — the production tree is
#: /mnt/projects/huncrc_dysplasia/quality_control and must stay untouched.
QC_OUTPUT_DIR = f"{PROJECT_PATH}/preprocessing/prov_test/qc"
DATASET_CSV_URI = (
    f"mlflow-artifacts:/212/{CREATE_DATASET_RUN_ID}/artifacts/dataset.csv"
)


def main() -> None:
    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov qc] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov qc] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to process"
        )

    # 2. the 2-slide input, straight from the create_dataset cut-down run
    dataset = pd.read_csv(download_artifacts(DATASET_CSV_URI))
    slides = dataset["slide_path"].to_list()
    print(f"[prov qc] {len(slides)} slides -> QC service: {slides}")

    output_path = Path(QC_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="🎭 QC (provenance, subset)") as run:
        # 3. original logic
        asyncio.run(
            qc_main(
                output_path=output_path,
                slides=slides,
                request_timeout=REQUEST_TIMEOUT,
                max_concurrent=MAX_CONCURRENT,
                qc_parameters=QC_PARAMETERS,
            )
        )

        # 4. the gate: the original exits green with missing slides — a
        # provenance run must not.
        metrics_path = output_path / "qc_metrics.csv"
        if not metrics_path.exists():
            raise SystemExit("[prov qc] qc_metrics.csv missing — QC produced nothing")
        metrics = pd.read_csv(metrics_path)
        missing = set(Path(s).stem for s in slides) - set(
            metrics["slide_name"].astype(str).str.zfill(3)
        )
        if missing:
            raise SystemExit(f"[prov qc] QC failed for slides {sorted(missing)} — not logging provenance")

        # 5. upload the mask tree (what tiling reads) + log provenance
        mlflow.log_artifacts(str(output_path), "qc_output")

        log_provenance(
            output=metrics,
            config={
                "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                "qc_parameters": QC_PARAMETERS,
                "request_timeout": REQUEST_TIMEOUT,
                "max_concurrent": MAX_CONCURRENT,
                "subset_slides": slides,
                "provenance": {"create_dataset_run_id": CREATE_DATASET_RUN_ID},
            },
            inputs=[f"runs:/{CREATE_DATASET_RUN_ID}"],
        )
        print(f"[prov qc] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
