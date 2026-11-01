#: Submit the FULL-COHORT hunCRC embedding-classifier training as a cluster
#: job (200 slides: the 186 + the 14 that re-passed QC).
#:
#: Trains ml.modeling.embeddings.DysplasiaEmbeddingsClassifier (MLP on
#: precomputed prov-gigapath tile embeddings) on tiling_200 (labels + split
#: column, all 200 slides) + preprocessing/embeddings/prov-gigapath (one
#: parquet per slide, 200/200 present). Same model / target as the 186-slide
#: run (experiment embeddings_huncrc) -- only the data paths and the
#: checkpoint dir (training/embeddings_huncrc200) differ, so the validation
#: AUCs are directly comparable.
#:
#:     uv run python -m ml +experiment=embeddings_huncrc200
#:
#: Tuning = hydra overrides via the env var, e.g.
#:     export HYDRA_OVERRIDES="++trainer.max_epochs=50 ++data.batch_size=512"
#:
#: NO data mount: the job reads only parquets (embeddings + tiling) from the
#: project share and writes checkpoints to
#: ${project_path}/training/embeddings_huncrc200.
#:
#: SIZING: 8 CPU feeds the 4 dataloader workers; 64Gi holds the three split
#: vector arrays at full-cohort scale -- 4.32 M tiles x 1536 x float32 ~=
#: 26.5 GB across train/val/test plus per-slide parquet read buffers
#: (largest slide ~1.6 GB). The 186-slide run (~15.5 GB of vectors) fit the
#: same 64Gi cap.
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
#: Distinct names let several GPU selectors race for a free slot (spec.template is
#: immutable, so a variant needs its own Job); delete the losers once one schedules.
JOB_NAME = os.environ.get(
    "JOB_NAME", "ulcerative-colitis-dysplasia-embeddings-huncrc200"
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
        f"uv run python -m ml +experiment=embeddings_huncrc200 {EXTRA_OVERRIDES}".rstrip(),
    ],
    storage=[storage.public.PROJECTS],
)
