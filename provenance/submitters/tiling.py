#: Submit the provenance cut-down of the tiling step as a cluster job.
#:
#: Runs provenance/tiling.py — a SEPARATE cut-down that reuses the original
#: preprocessing.tiling ray pipeline but leaves the original script untouched
#: (production reruns stay clean). Verifies the 200-slide registered cohort,
#: tiles the 2-slide test dataset (224 px @ 0.52 mpp, stride 112, the proven
#: pilot geometry) and logs the run with provenance — pinning the WHOLE chain
#: (create_dataset, split, QC, tissue) — into the isolated
#: "provenance_huncrc_test" experiment.
#:
#: WAVE 2 — submits AFTER the split/QC/tissue jobs finish: provenance/tiling.py
#: carries SPLIT_RUN_ID / QC_RUN_ID / TISSUE_RUN_ID constants that must be
#: filled in from the wave-1 job logs first, and the script refuses to run
#: while any of them is still a placeholder.
#:
#: SIZING. cpu=8/24Gi/shm=16Gi: mirrors the production tiling submitter —
#: ray workers + openslide + pyvips overlap computation; 2 slides keeps the
#: peak far below production, but the worker requests are unchanged.
#:
#: Storage: DATA (the .mrxs the workers open; verify_manifest also
#: fingerprints them) + PROJECTS (registration manifest + the per-slide
#: EMPIA annotation JSONs under /mnt/projects/huncrc_dysplasia/annotations).
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
    job_name="ulcerative-colitis-dysplasia-prov-tiling",
    username=USERNAME,
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
