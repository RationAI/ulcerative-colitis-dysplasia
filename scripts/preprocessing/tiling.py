#: Submit the full-cohort (all 200) TILING job for TCIA_HunCRC.
#:
#: Replaces the IBD template this file was (username "...", public=False,
#: storage.secure.DATA, a "+dataset=..." placeholder that made it unrunnable).
#: Same hunCRC setup as quality_control.py / tissue_masks.py / epithelium_masks.py.
#:
#: TILING CONSUMES FOUR ARTIFACTS, and every one of them is REQUIRED below on purpose:
#: the configs/dataset/processed*.yaml still carry the IBD cohort's URIs (experiment
#: 111), so a silent default would download the wrong cohort and fail mid-run.
#:   * QC (qc_metrics.csv + the 4 mask subdirs) -- preprocessing/tiling.py:338
#:     download_artifacts(config.dataset.mlflow_uris.qc). qc_agg does
#:     df.loc[<slide stem>] and KeyErrors on any slide missing, so the QC tree MUST be
#:     all 200 rows first. That is exactly what scripts/preprocessing/
#:     quality_control_retry.py uploads -- export the URI it prints as QC_ARTIFACT_URI.
#:   * TISSUE (the 200 tissue masks) -- :339 download_artifacts(.tissue). Also reused as
#:     the epithelium stand-in (configs/dataset/processed_w_masks.yaml: epithelium:
#:     ${mlflow_uris.tissue}); when the real episeg masks exist, add
#:     ++dataset.epithelium=<epi artifact URI> and the epithelium column goes live.
#:   * SPLITS -- train.csv / test_preliminary.csv / test_final.csv from a huncrc
#:     split_dataset run. train.csv carries the `fold` column tiling writes onto tiles.
#:     The three must be the DISJOINT 200-slide hunCRC split, not IBD's.
#:
#: TILING GEOMETRY is passed on the command line because configs/preprocessing/
#: tiling.yaml leaves mpp/tile_extent/stride as ??? (the job must say, so a wrong default
#: cannot silently retile the cohort): 224 px @ 0.52 mpp, stride 112 -- the proven
#: pilot values from scripts/run_tiling_pilot.py (level 2, tissue_threshold 0.5 is the
#: config default).
#:
#: MOUNTS: DATA (read_slides opens every whole slide itself: preprocessing/tiling.py:251)
#: + PROJECTS (annot_folder = ${project_path}/annotations = /mnt/projects/.../annotations,
#: the 200 EMPIA JSONs tiling tiles the dysplasia polygons from). Both are load-bearing.
#:
#: NOTE on size: download_artifacts(qc) pulls the whole qc_output (~8 GB, all 4 subdirs)
#: even though tiling only reads blur_per_pixel + artifacts_per_pixel. That is how the
#: stage is written (shared with IBD), so size the tmp to absorb it -- cpu=8/24Gi/16Gi shm
#: is the IBD template plus headroom for 545k x 588k pyramids + 200 mask downloads.
#:
#: Needs exported in the shell that runs this (all six; no safe defaults):
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
#:   export QC_ARTIFACT_URI=mlflow-artifacts:/168/<qc-rerun>/artifacts/qc_output
#:   export TISSUE_ARTIFACT_URI=mlflow-artifacts:/168/<tissue-run>/artifacts/tissue_masks
#:   export SPLIT_TRAIN_URI=mlflow-artifacts:/168/<split-run>/artifacts/train.csv
#:   export SPLIT_TEST_PRELIMINARY_URI=mlflow-artifacts:/168/<split-run>/artifacts/test_preliminary.csv
#:   export SPLIT_TEST_FINAL_URI=mlflow-artifacts:/168/<split-run>/artifacts/test_final.csv
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as the other huncrc submitters -- see quality_control.py for why the k8s
#: label and the MLflow identity differ, and why the token comes from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
#: (No DATASET_ARTIFACT_URI here on purpose: tiling reads the three split CSVs, not
#: dataset.csv, so there is nothing to default and nothing to get silently wrong.)

QC_ARTIFACT_URI = os.environ.get("QC_ARTIFACT_URI")
TISSUE_ARTIFACT_URI = os.environ.get("TISSUE_ARTIFACT_URI")
SPLIT_TRAIN_URI = os.environ.get("SPLIT_TRAIN_URI")
SPLIT_TEST_PRELIMINARY_URI = os.environ.get("SPLIT_TEST_PRELIMINARY_URI")
SPLIT_TEST_FINAL_URI = os.environ.get("SPLIT_TEST_FINAL_URI")

#: Proven pilot geometry (scripts/run_tiling_pilot.py): 224 px tiles @ 0.52 mpp, stride
#: 112, on level 2 of the hunCRC pyramid. Do not change without re-checking the mpp.
MPP = "0.52"
TILE_EXTENT = "224"
STRIDE = "112"


def _require(name: str, value: str | None, hint: str) -> str:
    if not value:
        raise SystemExit(f"{name} is unset: {hint}")
    return value


if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )
QC_ARTIFACT_URI = _require(
    "QC_ARTIFACT_URI",
    QC_ARTIFACT_URI,
    "run the QC retry job first (scripts/preprocessing/quality_control_retry.py); it "
    "prints mlflow-artifacts:/168/<run>/artifacts/qc_output when qc_metrics.csv hits "
    "200 rows. The 14-slide retry must finish before tiling, because tiling KeyErrors "
    "on any slide missing from qc_metrics.csv.",
)
TISSUE_ARTIFACT_URI = _require(
    "TISSUE_ARTIFACT_URI",
    TISSUE_ARTIFACT_URI,
    "export the 200-slide tissue-mask run as "
    "mlflow-artifacts:/168/<run>/artifacts/tissue_masks.",
)
SPLIT_TRAIN_URI = _require(
    "SPLIT_TRAIN_URI",
    SPLIT_TRAIN_URI,
    "export the huncrc split run's mlflow-artifacts:/168/<run>/artifacts/train.csv "
    "(run scripts/preprocessing/split_dataset.py first if no huncrc split exists yet).",
)
SPLIT_TEST_PRELIMINARY_URI = _require(
    "SPLIT_TEST_PRELIMINARY_URI",
    SPLIT_TEST_PRELIMINARY_URI,
    "export the split run's test_preliminary.csv artifact URI.",
)
SPLIT_TEST_FINAL_URI = _require(
    "SPLIT_TEST_FINAL_URI",
    SPLIT_TEST_FINAL_URI,
    "export the split run's test_final.csv artifact URI.",
)

submit_job(
    job_name="ulcerative-colitis-dysplasia-tiling-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="24Gi",
    shm="16Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        "uv run python -m preprocessing.tiling +dataset=processed_w_masks "
        f"++dataset.mlflow_uris.qc={QC_ARTIFACT_URI} "
        f"++dataset.mlflow_uris.tissue={TISSUE_ARTIFACT_URI} "
        f"++dataset.mlflow_uris.splits.train={SPLIT_TRAIN_URI} "
        f"++dataset.mlflow_uris.splits.test_preliminary={SPLIT_TEST_PRELIMINARY_URI} "
        f"++dataset.mlflow_uris.splits.test_final={SPLIT_TEST_FINAL_URI} "
        f"mpp={MPP} tile_extent={TILE_EXTENT} stride={STRIDE} "
        "project_path=/mnt/projects/huncrc_dysplasia",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
