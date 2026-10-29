#: Submit the project-dataset registration as a cluster job.
#:
#: Runs scripts/provenance/register_dataset.py: registers the hunCRC cohort
#: (200 raw slides, preprocessing/dataset_split.csv) into the project's
#: "Ulcerative Colitis Dysplasia" experiment — one run per dataset version,
#: pinning per-file size+mtime fingerprints. Writes the canonical
#: manifest.csv next to the cohort csv; the data itself never moves.
#:
#: SUBMIT THIS ONLY AFTER the create-user job has FINISHED — capture_environment
#: looks up the user in User_Registry and fails hard otherwise.
#:
#: Storage: DATA (read-only, stat the .mrxs files) + PROJECTS (read the cohort
#: csv, write manifest.csv).
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
import os

from kube_jobs import storage, submit_job

BRANCH = "huncrc"
#: The Rancher username — exported into the job as JOB_USER; this exact
#: string is what every provenance lookup keys on.
USERNAME = "borisim"

#: Same values as the other huncrc submitters -- see embeddings_huncrc.py.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-prov-register-dataset",
    username=USERNAME,
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
        f"export JOB_USER='{USERNAME}'",
        "uv run --no-project "
        "--with 'rationai-provenance @ git+https://github.com/RationAI/provenance.git' "
        "python -m scripts.provenance.register_dataset",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
