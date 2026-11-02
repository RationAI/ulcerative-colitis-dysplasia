#: Submit the provenance cut-down of the tiling step as a cluster job.
#:
#: Runs provenance/tiling.py — a SEPARATE cut-down that leaves the original
#: preprocessing.tiling script untouched (production reruns stay clean).
#: REUSE MODE: both test slides are already tiled in the production full-cohort
#: tiling (preprocessing/tiling_200), so instead of running ray + openslide it
#: filters tiling_200 to exactly the 2 chain slides (predicate pushdown —
#: only their tile rows are loaded), RESTAMPS split/fold to the chain's own
#: split (train=059, test_final=155), writes the per-split parquet layout the
#: tile_masks cut-down consumes, and logs the run with provenance — the reuse
#: recorded explicitly (run name, cfg_masks_source, cfg_reuse_reason) — into
#: the isolated "provenance_huncrc_test" experiment.
#:
#: FAST: pandas/pyarrow only, no ray/openslide — minutes. No TISSUE_RUN_ID
#: pin: reuse mode consumes the split CSVs + the production share, not the
#: chain's QC/tissue artifacts (those runs stay pinned by the tile_masks and
#: training runs, which carry the full ancestor list).
#:
#: SIZING. cpu=4/16Gi: predicate-pushed reads load only ~177k tile rows;
#: the 16 GiB headroom is for the pandas concat + parquet writes.
#:
#: Storage: DATA (verify_manifest fingerprints the .mrxs) + PROJECTS
#: (registration manifest + the read-only tiling_200 source dir). The job
#: writes nothing to the project share.
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
JOB_NAME = os.environ.get("JOB_NAME", "ulcerative-colitis-dysplasia-prov-tiling")


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
    memory="16Gi",
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
        "uv run --no-sync python -m provenance.tiling",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
