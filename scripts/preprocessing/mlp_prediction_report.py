#: Submit the MLP val-set prediction job (eval + predict + epi + upload) as a
#: cluster job.
#:
#: Runs scripts/make_mlp_prediction_report.py --eval --predict --epi --upload:
#:   * --eval    the checkpoint's val AUC/acc on the filtered val set (printed
#:               first, so the job log records the number the report quotes)
#:   * --predict per-tile dysplasia probability -> level-3 tiff per val slide,
#:               written to a NEW dir preprocessing/mlp_predictions/
#:   * --epi     raw epi masks shrunk 2x -> preprocessing/epithelium_masks/level3
#:               (NEW dir; resumable)
#:   * --upload  both into the report run (45a957c5..., the 186 tile-masks
#:               report's artifact host; annotation + outline layers already
#:               live there)
#:
#: The report page itself is NOT built here: report-fast is not in uv.lock, so
#: the job cannot import it. After this job lands, run
#: `python scripts/make_mlp_prediction_report.py --report /tmp/...html
#: --publish` on the notebook pod (light: artifact listing + two ~1 MB html
#: uploads).
#:
#: Why the job and not the pod: each val slide's embedding parquet is ~1.5 GB
#: of list<double>; the pod is 32 GB and died twice on the interactive
#: attempts.
#:
#: Storage:
#:   * public.PROJECTS -- reads tiling_200 + embeddings + the checkpoint +
#:     annotation/outline/epi masks, writes the two NEW mask dirs. Read-write.
#:   * public.DATA -- predict opens each .mrxs for the level-3 dimensions and
#:     mpp (ratiopath.openslide).
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
#:
#: The job clones the repo from GitHub at BRANCH below, so the script must be
#: pushed first.
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as the other hunCRC submitters -- token from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

#: Single-process, one slide at a time: per-slide peak is the ~1.5 GB parquet
#: buffer + 0.8 GB float32 matrix + torch transient, so 16 GiB is generous.
#: cpu=8 matches the other hunCRC submitters.
submit_job(
    job_name="ulcerative-colitis-dysplasia-mlp-prediction-report-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="16Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        "uv run python scripts/make_mlp_prediction_report.py "
        "--eval --predict --epi --upload",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
