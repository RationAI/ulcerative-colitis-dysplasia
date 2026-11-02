#: Submit the provenance cut-down of the split-dataset step as a cluster job.
#:
#: Runs provenance/split_dataset.py — a SEPARATE cut-down that reuses
#: the original preprocessing.split_dataset logic but leaves the original
#: script untouched (production reruns stay clean). Verifies the 200-slide
#: registered cohort, splits the 2-slide test dataset (0.7/0.15/0.15, 5
#: folds, rs=42) and logs the run with provenance (env, config, split sizes,
#: pinned input run) into the isolated "provenance_huncrc_test" experiment.
#:
#: INSTANT: pure pandas on 2 rows — no imaging, no service calls.
#:
#: Storage: DATA (verify_manifest fingerprints the .mrxs) + PROJECTS (read
#: the registration manifest). The job writes nothing to the project share.
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
import os

from kube_jobs import storage, submit_job

BRANCH = "huncrc"
#: The Rancher username — k8s label + provenance identity (exported as JOB_USER
#: into the job; the provenance lib resolves the user from it and fails if unset).
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
    job_name="ulcerative-colitis-dysplasia-prov-split-dataset",
    username=USERNAME,
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=2,
    memory="8Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        # Provenance: capture_environment resolves the user from JOB_USER and
        # fails the run when it is unset or unregistered (kube-jobs 0.4.0 does
        # not inject it).
        f"export JOB_USER='{USERNAME}'",
        "uv sync --frozen",
        # NOT `uv run --with`: that installs into an overlay env ranked behind
        # the project venv, and the project's `rationai` package (a regular
        # package) shadows `rationai.provenance` -- the 2026-10-29 create_dataset
        # job died on ModuleNotFoundError despite the overlay building fine.
        # Install into the venv so the files merge into site-packages/rationai/,
        # then run without re-syncing it away (pyproject/uv.lock stay pristine).
        "uv pip install --python .venv/bin/python "
        "'rationai-provenance @ git+https://github.com/RationAI/provenance.git'",
        "uv run --no-sync python -m provenance.split_dataset",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
