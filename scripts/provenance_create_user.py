#: Submit the one-time "register yourself" provenance step as a cluster job.
#:
#: Runs scripts/provenance/create_user.py, which records who you are in the
#: User_Registry experiment — once, ever (idempotent). Every later
#: provenance-logged run looks you up by exactly this username; runs with an
#: unregistered user fail at startup.
#:
#: Requires the UPDATED village kube-jobs (no username= arg — it derives
#: JOB_USER from your kube config and injects it into the job env).
#:
#: Job env: create_user.py needs JOB_USER (injected) + MLflow creds (exported
#: below as in the other huncrc submitters). No data mount — only MLflow.
import os

from kube_jobs import submit_job

BRANCH = "huncrc"

#: Same values as the other huncrc submitters -- see embeddings_huncrc.py.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-create-user",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=1,
    memory="4Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        # TODO: simplify to `uv sync --frozen` + `uv run python -m ...` once
        # rationai-provenance is a pinned project dependency (uv add).
        "uv run --no-project "
        "--with 'rationai-provenance @ git+https://github.com/RationAI/provenance.git' "
        "python -m scripts.provenance.create_user",
    ],
)
