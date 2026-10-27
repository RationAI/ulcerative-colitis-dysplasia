#: Submit the hunCRC embedding-classifier training as a cluster job.
#:
#: Trains ml.modeling.embeddings.DysplasiaEmbeddingsClassifier (MLP on
#: precomputed prov-gigapath tile embeddings) on the SAME artifacts the
#: U-Net experiment uses for its targets -- tiling_186_merged (labels +
#: split column) + embeddings_186_v3/prov-gigapath -- and shares its
#: binary dysplasia target, so the two are directly comparable.
#:
#:     uv run python -m ml +experiment=embeddings_huncrc
#:
#: Tuning = hydra overrides via the env var, e.g.
#:     export HYDRA_OVERRIDES="++trainer.max_epochs=50 ++data.batch_size=512"
#:
#: NO data mount: the job reads only parquets (embeddings + tiling) from the
#: project share and writes checkpoints to ${project_path}/embeddings_huncrc.
#:
#: SIZING: 8 CPU feeds the 8 dataloader workers (each LRU-caches whole-slide
#: vector arrays, ~0.1-0.3 GiB); 16Gi holds 8 x that + the model. The MLP is
#: tiny -- the GPU mostly saves the epoch time on ~2.5M tiles.
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as the other huncrc submitters -- see unet_dysplasia.py.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
#: JobGPUType from kube_jobs.resolve_gpu: "A100" | "A40" | "H100" | ...
GPU = os.environ.get("GPU", "A100")
EXTRA_OVERRIDES = os.environ.get("HYDRA_OVERRIDES", "")


if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-embeddings-huncrc",
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
        f"uv run python -m ml +experiment=embeddings_huncrc {EXTRA_OVERRIDES}".rstrip(),
    ],
    storage=[storage.public.PROJECTS],
)
