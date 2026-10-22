#: Submit the dysplasia U-Net SEGMENTATION experiment as a cluster job.
#:
#: EXPERIMENT, not part of the main pipeline: trains ml/unet (tile-level
#: U-Net segmentation of LGD/HGD regions) on the SAME artifacts the main
#: embedding pipeline consumes -- tiling_186_merged (level-2 tile grid + slide
#: paths) and annotation_masks/ (level-5 per-class mask pyramids) -- but writes
#: nothing back into it. Its output (checkpoints + demo_val.png) lands in
#: ${project_path}/unet_dysplasia. configs/unet.yaml carries every parameter
#: with a default, so the job runs the plain entrypoint:
#:
#:     uv run python -m ml.unet.train
#:
#: Tuning = hydra overrides appended to that line, e.g.
#:     uv run python -m ml.unet.train ++trainer.max_epochs=20 ++data.batch_size=8
#:
#: PREREQUISITES: (1) tiling_186_merged exists (merge_tiling_shards.py ran);
#: (2) annotation_masks/ exists with lowgrade_dysplasia/ + highgrade_dysplasia/
#: (the annotation_masks stage ran); (3) libopenslide + libvips are in the base
#: image (both proven: tiling/tile-masks and annotation_masks jobs use them).
#:
#: GPU: first GPU job in this project -- submit_job(gpu=...) selects the node
#: (default A100-80GB, override with export GPU=A40 / H100 / "Tesla P100").
#: torch 2.11 in the committed lock is the CUDA wheel; cerit.io/rationai/base
#: is the same image the prov-gigapath GPU service runs, so CUDA works.
#:
#: MOUNTS: DATA (OpenSlide reads every whole .mrxs) + PROJECTS (tiling +
#: masks in, checkpoints out).
#:
#: SIZING: cpu=8 feeds the 8 dataloader workers (each opens an mrxs + two
#: mask TIFs, LRU 2 slides); 32Gi holds 8 x ~2-4 GiB worker state + the
#: A100-side model. Batch 16 x 512px is small for an A100; raise
#: data.batch_size via override if GPU utilization says so.
#:
#: RUNTIME: ~4k pos + 4k neg patches/epoch, 10 epochs -> a couple of hours on
#: one A100. No resume (fits in the job TTL); a rerun is just resubmitting.
#:
#: Needs exported in the shell that runs this (MLflow only -- the rationai
#: Trainer autologs metrics; in-cluster URI, so no URL, just the identity):
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as the other huncrc submitters -- see quality_control.py for why
#: the k8s label and the MLflow identity differ, and why the token comes from
#: the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
#: JobGPUType from kube_jobs.resolve_gpu: "A100" | "A40" | "H100" |
#: "Tesla P100" | "mig-1g.10gb" | "mig-2g.20gb".
GPU = os.environ.get("GPU", "A100")
EXTRA_OVERRIDES = os.environ.get("HYDRA_OVERRIDES", "")


if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-unet-dysplasia-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="32Gi",
    gpu=GPU,
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        f"uv run python -m ml.unet.train {EXTRA_OVERRIDES}".rstrip(),
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
