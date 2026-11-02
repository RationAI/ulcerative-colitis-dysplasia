#: Provenance cut-down of the tissue-masks step.
#:
#: Tiling REQUIRES real tissue masks (its tissue gate), so this is a
#: mandatory part of the chain even though it is a separate pipeline stage.
#:
#: The original preprocessing/tissue_masks.py is left UNTOUCHED. This copy
#: reuses its process_slide() ray actor (one source of truth for the logic)
#: on a 2-slide subset and wraps it in a provenance-logged MLflow run:
#:
#:   1. verify_manifest — the registered cohort still matches the pinned run
#:   2. process_slide   — the original logic, 2 slides (local pyvips/openslide
#:                        segmentation — minutes, NOT the QC service)
#:   3. log_provenance  — env + hardware, config params, slide stats +
#:                        the pinned input run (runs:/<create_dataset run id>)
#:
#: Run as a kube-job (submitter: provenance/submitters/tissue_masks.py; needs
#: JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.tissue_masks
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import ray
from mlflow.artifacts import download_artifacts
from rationai.masks import process_items
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest

#: Reuse the original logic — the original module stays byte-identical.
from preprocessing.tissue_masks import process_slide

#: Where the cut-down test runs live, isolated from the production runs.
EXPERIMENT = "provenance_huncrc_test"
#: Pinned create_dataset cut-down run (provenance/create_dataset.py) — its
#: dataset.csv artifact IS the 2-slide input for this whole test chain.
CREATE_DATASET_RUN_ID = "bc5160ba96064230b375cb15fdf78320"
#: Pinned registration run of the huncrc cohort (verify gate).
DATASET_RUN_ID = "4ef2fb98adb8432b9ee224fc5bdd5231"

#: Same values as configs/preprocessing/tissue_masks.yaml.
LEVEL = 3
MAX_CONCURRENT = 4

PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
DATASET_CSV_URI = (
    f"mlflow-artifacts:/212/{CREATE_DATASET_RUN_ID}/artifacts/dataset.csv"
)


def main() -> None:
    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov tissue] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov tissue] dataset verification FAILED (run {DATASET_RUN_ID}) — refusing to process"
        )

    # 2. the 2-slide input, straight from the create_dataset cut-down run
    dataset = pd.read_csv(download_artifacts(DATASET_CSV_URI))
    slides = dataset["slide_path"].to_list()
    print(f"[prov tissue] {len(slides)} slides: {slides}")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="🧻 Tissue masks (provenance, subset)") as run:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 3. original logic
            with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
                process_items(
                    slides,
                    process_item=process_slide,
                    fn_kwargs={"level": LEVEL, "output_path": Path(tmpdir)},
                    max_concurrent=MAX_CONCURRENT,
                )

            # 4. upload the masks (what tiling reads) + log provenance
            mlflow.log_artifacts(tmpdir, "tissue_masks")

            log_provenance(
                output=dataset,
                config={
                    "dataset": {"name": "huncrc-cohort", "version": "1.0.0"},
                    "level": LEVEL,
                    "max_concurrent": MAX_CONCURRENT,
                    "subset_slides": [Path(s).stem for s in slides],
                    "provenance": {"create_dataset_run_id": CREATE_DATASET_RUN_ID},
                },
                inputs=[f"runs:/{CREATE_DATASET_RUN_ID}"],
            )
        print(f"[prov tissue] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
