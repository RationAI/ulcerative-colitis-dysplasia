#: Provenance cut-down of the dataset-creation step.
#:
#: The original preprocessing/create_dataset.py is left UNTOUCHED — it stays
#: ready for clean production reruns. This copy reuses its core
#: create_dataset() function (one source of truth for the logic, zero
#: duplication) and wraps it in a provenance-logged MLflow run:
#:
#:   1. verify_manifest — all 200 registered slides on disk still match the
#:      pinned registration run (job fails on drifted data)
#:   2. create_dataset  — the original logic (cheap: glob + regex, no imaging)
#:   3. subset          — only SUBSET_N rows become the test output
#:   4. log_provenance  — env + hardware, config params, output stats +
#:      file manifest, and the pinned input run (runs:/<dataset run id>)
#:
#: Run as a kube-job (submitter: provenance/submitters/create_dataset.py; needs
#: JOB_USER + MLflow credentials, both exported there):
#:
#:     uv run python -m provenance.create_dataset
import os
import tempfile

import mlflow
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest

#: Reuse the original logic — the original module stays byte-identical.
from preprocessing.create_dataset import create_dataset

#: Where the cut-down test runs live, isolated from the production runs.
#: (Input pins work across experiments — they are run ids, not experiment ids.)
EXPERIMENT = "provenance_huncrc_test"
#: Pinned registration run of the huncrc cohort (provenance/register_dataset.py).
#: Never "latest" — re-pin after any re-registration.
DATASET_RUN_ID = "21bb67f96ea446b8b70ab3ac86710edd"
#: Rows of the test output (verify still covers all 200 registered files).
SUBSET_N = 2

#: Same values as configs/dataset/raw.yaml + configs/base.yaml — hardcoded here
#: because this script intentionally does not go through the hydra config tree.
PROJECT_PATH = "/mnt/projects/huncrc_dysplasia"
SLIDES_PATH = "/mnt/data/Public/Public/colorectum/TCIA_HunCRC/PKG - Hungarian-Colorectal-Screening_v2"
TILES_PATH = "/mnt/data/Public/Public/colorectum/TCIA_HunCRC/PKG - Hungarian-Colorectal-Screening_ML-dataset/ML_dataset"
PATTERN = r"^\d{3}\.mrxs$"


def main() -> None:
    # 1. verify the registered cohort before touching anything
    manifest_path = os.path.join(PROJECT_PATH, "preprocessing", "manifest.csv")
    check = verify_manifest(manifest_path, os.path.dirname(manifest_path), DATASET_RUN_ID)
    for detail in check["details"]:
        print(f"[prov create_dataset] verify: {detail}")
    if not check["verified"]:
        raise SystemExit(
            f"[prov create_dataset] dataset verification FAILED (run {DATASET_RUN_ID}) "
            "— refusing to process"
        )

    # 2. original logic on the full cohort (glob + regex only — cheap)
    dataset, missing_tiles = create_dataset(SLIDES_PATH, TILES_PATH, PATTERN)

    # 3. cut down the test output
    output = dataset.head(SUBSET_N)
    print(f"[prov create_dataset] {len(dataset)} slides listed, "
          f"{len(output)} in test output, {len(missing_tiles)} missing tile labels")

    # 4. provenance-logged run
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="📂 Dataset Creation (provenance, subset)") as run:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "dataset.csv")
            output.to_csv(csv_path, index=True)
            mlflow.log_artifact(csv_path)

            if missing_tiles:
                missing_path = os.path.join(tmpdir, "missing_tiles.txt")
                with open(missing_path, "w") as f:
                    f.write("\n".join(missing_tiles) + "\n")
                mlflow.log_artifact(missing_path)

        log_provenance(
            output=output,
            config={
                "dataset": {
                    "name": "huncrc-cohort",
                    "version": "1.0.0",
                    "slides_path": SLIDES_PATH,
                    "tiles_path": TILES_PATH,
                    "subset_n": SUBSET_N,
                },
                "provenance": {"dataset_run_id": DATASET_RUN_ID},
            },
            inputs=[f"runs:/{DATASET_RUN_ID}"],
        )
        print(f"[prov create_dataset] done — run {run.info.run_id} in experiment {EXPERIMENT}")


if __name__ == "__main__":
    main()
