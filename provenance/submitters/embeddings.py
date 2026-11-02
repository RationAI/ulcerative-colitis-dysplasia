#: Submit the provenance cut-down of the embeddings step as a cluster job.
#:
#: Runs provenance/embeddings.py — a SEPARATE cut-down that leaves the
#: original preprocessing.embeddings script untouched. REUSE MODE: the
#: prov-gigapath embeddings for the 2 test slides (155/059) already exist on
#: the project share (preprocessing/embeddings/prov-gigapath, one parquet per
#: slide, row counts matching the tiling_200 grid 1:1), so instead of burning
#: GPU hours on gigapath inference it copies their parquets into a
#: prov-dedicated staging dir, GATES on both files present AND row counts
#: matching the tiling_200 grid (the same check the training dataset enforces),
#: records the reuse explicitly in the run (name, cfg_masks_source,
#: cfg_reuse_reason) and logs it into the isolated "provenance_huncrc_test"
#: experiment.
#:
#: FAST: copy ~170 MB + a predicate-pushed grid check. Minutes.
#:
#: Storage: DATA (verify_manifest fingerprints the .mrxs) + PROJECTS
#: (registration manifest + the read-only prov-gigapath source dir + the
#: prov-dedicated staging dir preprocessing/prov_test/embeddings — the
#: production embeddings/ tree is never touched).
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
#: Distinct names let a resubmit go out renamed instead of deleting the old job.
JOB_NAME = os.environ.get("JOB_NAME", "ulcerative-colitis-dysplasia-prov-embeddings")


if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name=JOB_NAME,
    username=USERNAME,
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=4,
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
        "uv run --no-sync python -m provenance.embeddings",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
