#: Submit the provenance cut-down of the tile-masks step as a cluster job.
#:
#: Runs provenance/tile_masks.py — a SEPARATE cut-down that reuses the
#: original preprocessing.tile_masks ray logic but leaves the original
#: script untouched (production reruns stay clean). Verifies the 200-slide
#: registered cohort, builds the per-tile percentage masks + outlines for
#: the 2-slide test dataset (level 3) and logs the run with provenance into
#: the isolated "provenance_huncrc_test" experiment.
#:
#: RUNS LAST in the chain — submits AFTER the tiling job finishes:
#: provenance/tile_masks.py carries a TILING_RUN_ID constant that must be
#: filled in from the tiling job logs first, and the script refuses to run
#: while it is still a placeholder.
#:
#: SIZING. cpu=4/96Gi: mirrors the production tile_masks submitter —
#: MaskBuilder accumulates full tile spans per percentage column, so the
#: per-slide peak memory dominates; 2 slides at max_concurrent 2 stay far
#: below it, but the request is unchanged.
#:
#: Storage: DATA (the .mrxs the tasks open; verify_manifest also
#: fingerprints them) + PROJECTS (read the registration manifest).
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
JOB_NAME = os.environ.get("JOB_NAME", "ulcerative-colitis-dysplasia-prov-tile-masks")

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
    memory="96Gi",
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
        "uv run --no-sync python -m provenance.tile_masks",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
