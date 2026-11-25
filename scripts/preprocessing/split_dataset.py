#: Submit split_dataset for TCIA_HunCRC as a cluster job.
#:
#: Reads the dataset.csv written by create_dataset and uploads train/test splits.
#: Same hunCRC setup as scripts/preprocessing/create_dataset.py -- public cloud,
#: BRANCH pinned, MLflow token exported into the job -- with two differences:
#:
#:   * No storage mount. The payload never touches disk: input comes from MLflow
#:     (preprocessing/split_dataset.py:51, download_artifacts) and the three outputs
#:     go back to MLflow (lines 59-67, tmpdir -> logger.log_artifact). Omitting
#:     storage= gives zero volumes (verified: kube_jobs Storage() -> [], []).
#:   * +dataset=processed, which inherits hunCRC's raw config, plus a ++ override of
#:     dataset.mlflow_uris.dataset. processed.yaml:6 hardcodes an IBD artifact URI
#:     (experiment 111). Overridden here rather than in the config because configs
#:     come from the GitHub clone, so an edit would need a commit + push; the script
#:     list is baked into the job spec at submit time.
#:
#: Needs two things exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
#:   export DATASET_ARTIFACT_URI=mlflow-artifacts:/<exp>/<run>/artifacts/dataset.csv
#: The URI is on the create_dataset run's page in the MLflow UI, Artifacts tab.
import os

from kube_jobs import submit_job


BRANCH = "huncrc"

#: Same values as create_dataset.py -- see the notes there for why the k8s label and
#: the MLflow identity differ, and why the token comes from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
#: dataset.csv uploaded by the hunCRC create_dataset run (experiment 168). Override with
#: `export DATASET_ARTIFACT_URI=...` after a later create_dataset run; scripts/
#: find_dataset_artifact.py prints the URI for the newest one.
DATASET_ARTIFACT_URI = os.environ.get(
    "DATASET_ARTIFACT_URI",
    "mlflow-artifacts:/168/2ca0c1928c2c48acb56a5110efefb957/artifacts/dataset.csv",
)

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )
if not DATASET_ARTIFACT_URI:
    raise SystemExit(
        "DATASET_ARTIFACT_URI is unset. Open the successful create_dataset run in the "
        "MLflow UI, copy dataset.csv's artifact URI off the Artifacts tab, and export it "
        "as DATASET_ARTIFACT_URI (looks like mlflow-artifacts:/<exp>/<run>/artifacts/"
        "dataset.csv). Without this the job reads the IBD experiment 111 dataset baked "
        "into configs/dataset/processed.yaml and splits the wrong cohort."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-split-dataset-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=2,
    memory="4Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        "uv run python -m preprocessing.split_dataset +dataset=processed "
        f"++dataset.mlflow_uris.dataset={DATASET_ARTIFACT_URI}",
    ],
)
