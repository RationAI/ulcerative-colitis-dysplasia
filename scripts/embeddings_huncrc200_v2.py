#: Submit the FULL-COHORT hunCRC embedding-classifier training as a cluster
#: job -- LABEL POLICY v2 (doc.md section 7).
#:
#: Same job shape as scripts/embeddings_huncrc200.py; the experiment
#: embeddings_huncrc200_v2 points the dataset at the v2 training table
#: (preprocessing/training_tiles, built by scripts/make_training_tiles.py)
#: instead of the legacy all-tiles convention. The table is ~35% of the
#: legacy sample count (1.54 M tiles vs 4.32 M), so the 64Gi cap is
#: headroom, not a requirement.
#:
#:     uv run python -m ml +experiment=embeddings_huncrc200_v2
#:
#: Tuning = hydra overrides via the env var, e.g.
#:     export HYDRA_OVERRIDES="++trainer.max_epochs=50 ++data.batch_size=512"
#:
#: NO data mount: the job reads only parquets (embeddings + tiling + the
#: v2 table) from the project share and writes checkpoints to
#: ${project_path}/training/embeddings_huncrc200_v2.
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
JOB_NAME = os.environ.get(
    "JOB_NAME", "ulcerative-colitis-dysplasia-embeddings-huncrc200-v2"
)
EXTRA_OVERRIDES = os.environ.get("HYDRA_OVERRIDES", "")


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
        "uv sync --frozen",
        f"uv run python -m ml +experiment=embeddings_huncrc200_v2 {EXTRA_OVERRIDES}".rstrip(),
    ],
    storage=[storage.public.PROJECTS],
)
