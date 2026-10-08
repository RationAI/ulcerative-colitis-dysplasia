#: Submit the 3-slide tile-mask pilot for TCIA_HunCRC as a cluster job.
#:
#: WHY A JOB. The pilot (scripts/run_tile_masks_pilot.py) is the only thing that can
#: run this stage for this cohort: preprocessing/tile_masks.py reads MLflow URIs from
#: the IBD config, and this cohort's tiling lives on the project share as parquet
#: written by scripts/run_tiling_pilot.py. The notebook pod (32 GiB cgroup) OOM-kills
#: at level 3: one MaskBuilder float32 canvas is 73600x68192 px = ~20 GiB, and at
#: level 2 (147200x136384) it is ~80 GiB. A 64 Gi job runs level 3 with headroom.
#:
#: LEVEL 3, not 2, on purpose: IBD's tile masks are also 2x coarser than its tiling
#: (level-3 masks over 0.52-mpp tiling), so level 3 matches the pipeline convention
#: and keeps each output TIFF at ~5 GiB instead of 20.
#:
#: MOUNTS: both, and both are load-bearing.
#:   * public.DATA    -- the pilot opens each .mrxs (OpenSlide in
#:     preprocessing/tile_masks.py:29) to read level_dimensions/slide_resolution.
#:   * public.PROJECTS -- reads /mnt/projects/huncrc_dysplasia/tiling_pilot/{slides,
#:     tiles} parquet and writes /mnt/projects/huncrc_dysplasia/tile_masks_pilot.
#:
#: NO MLFLOW CREDENTIALS: the pilot never talks to MLflow -- no autolog, no artifacts;
#: it reads/writes the share directly, so unlike the other submitters there is no
#: token to export.
#:
#: The job clones this file's code from GitHub at BRANCH, so the pilot and any
#: preprocessing/ edits must be committed and pushed before running this.
#:
#: --workers 1 on purpose: one 20-GiB canvas at a time keeps the peak in a single
#: worker; with 64 Gi total there is headroom, but the masks give nothing for speed
#: here (3 slides, numpy-bound).
from kube_jobs import storage, submit_job


BRANCH = "huncrc"

submit_job(
    job_name="ulcerative-colitis-dysplasia-tile-masks-huncrc-pilot",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="64Gi",
    shm="16Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python scripts/run_tile_masks_pilot.py --level 3 --workers 1",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
