#: Submit the provenance cut-down of the TRAINING step as a cluster job.
#:
#: Runs provenance/training.py — the real compute leg of the test chain. It
#: runs the ORIGINAL training entry point (ml module: DysplasiaEmbeddings
#: Classifier on precomputed tile embeddings, @hydra.main + @autolog) on the
#: 2-slide subset, in-process, using the UNMODIFIED experiment config
#: (+experiment=embeddings_huncrc200; the prov deviations are CLI overrides —
#: no config file is added or modified), checkpoints to the prov-dedicated
#: dir training/embeddings_huncrc2prov, and appends the provenance log to the
#: SAME run the training creates — pinning the FULL ancestor chain
#: (registration, create_dataset, split, QC, tissue, tiling, embeddings).
#:
#: SLOW: real GPU training, 2 epochs over ~177k tile vectors (A100) —
#: the only compute step of the chain besides tile_masks.
#:
#: WAVE 3 — submits AFTER the tiling + embeddings jobs finish:
#: provenance/training.py carries TILING_RUN_ID / EMBEDDINGS_RUN_ID constants
#: that must be filled in from those job logs first, and the script refuses
#: to run while either is still a placeholder.
#:
#: SIZING: mirrors the production training submitter (8 CPU feeds the 4
#: dataloader workers; 64Gi is far more than the 2-slide vector arrays need,
#: but the dataloader + parquet read buffers follow the production sizing).
#:
#: Storage: DATA (verify_manifest fingerprints the .mrxs) + PROJECTS
#: (registration manifest + the read-only tiling_200 / prov-gigapath sources
#: + the prov-dedicated checkpoint dir). The production
#: training/embeddings_huncrc200 is never touched.
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
import os

from kube_jobs import storage, submit_job

BRANCH = "huncrc"

#: Same values as the other hunCRC submitters -- token from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
#: JobGPUType from kube_jobs.resolve_gpu: "A100" | "A40" | "H100" | ...
GPU = os.environ.get("GPU", "A100")
#: Distinct names let several GPU selectors race for a free slot (spec.template
#: is immutable, so a variant needs its own Job); delete the losers once one
#: schedules.
JOB_NAME = os.environ.get("JOB_NAME", "ulcerative-colitis-dysplasia-prov-training")


if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name=JOB_NAME,
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="64Gi",
    gpu=GPU,
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        # Provenance: capture_environment resolves the user from JOB_USER and
        # fails the run when it is unset or unregistered (kube-jobs 0.4.0 does
        # not inject it).
        "export JOB_USER='borisim'",
        "uv sync --frozen",
        # NOT `uv run --with`: that installs into an overlay env ranked behind
        # the project venv, and the project's `rationai` package (a regular
        # package) shadows `rationai.provenance` -- the 2026-10-29 create_dataset
        # job died on ModuleNotFoundError despite the overlay building fine.
        # Install into the venv so the files merge into site-packages/rationai/,
        # then run without re-syncing it away (pyproject/uv.lock stay pristine).
        "uv pip install --python .venv/bin/python "
        "'rationai-provenance @ git+https://github.com/RationAI/provenance.git'",
        "uv run --no-sync python -m provenance.training",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
