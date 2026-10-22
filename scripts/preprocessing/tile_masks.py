#: Submit the full-cohort TILE-MASKS job for TCIA_HunCRC.
#:
#: Rasterises the merged v3 tiling into per-slide, per-column level-2 TIFF
#: pyramids via scripts/run_tile_masks_pilot.py (which calls the actual
#: preprocessing/tile_masks.py process_slide, unchanged): one mask set per
#: percentage column -- epithelium (REAL epi % after
#: scripts/merge_epithelium_column.py replaced the tissue stand-in), blur,
#: artifacts, lowgrade_dysplasia, highgrade_dysplasia -- plus the tile-boundary
#: outlines mask. 186 slides x 6 = 1,116 TIFFs under
#: /mnt/projects/huncrc_dysplasia/tile_masks_186/{col}/{stem}.tiff.
#:
#: PREREQUISITES, in order: (1) scripts/merge_tiling_shards.py has run
#: (tiling_186_merged/{slides,tiles}); (2) the epi column has been merged into
#: it (scripts/merge_epithelium_column.py) -- otherwise the epithelium/ masks
#: would show tissue scores again.
#:
#: WHY A JOB AND NOT THE POD: 186 whole slides opened sequentially, hours of
#: runtime; the pod can sleep/restart, a job pod cannot. Same code either way
#: (the pilot script's --tiling-dir/--out flags make the pod path identical).
#:
#: MOUNTS: DATA (process_slide opens every whole slide via OpenSlide) +
#: PROJECTS (merged tiling in, mask TIFFs out). No MLflow: the masks live on
#: the share; upload later if the xOpat/report path needs them as artifacts.
#:
#: SIZING: the pilot reserves 12Gi per task and --workers 2 runs two slides at
#: once; the first submit (16Gi) had Ray killing workers at 95% of the pod
#: memory (15.23GB / 16.00GB, top user ray::process_slide at 10.9GB) and
#: retrying forever without writing a single TIFF. Four level-2 float32
#: MaskBuilder accumulators plus the OpenSlide handle are ~11 GB per slide, so
#: 2 x 11 + driver + Ray overhead needs ~30Gi -- 48Gi carries the peaks.
#:
#: RESUME: yes -- an existing {out}/{col}/{stem}.tiff is accepted as done for
#: that slide+column, so a resubmission after a crash continues where it left
#: off (re-run the whole submitter; no state to export).
#:
#: Needs nothing exported -- no MLflow in this job.
from kube_jobs import storage, submit_job


BRANCH = "huncrc"
TILING_DIR = "/mnt/projects/huncrc_dysplasia/tiling_186_merged"
OUT_DIR = "/mnt/projects/huncrc_dysplasia/tile_masks_186"

submit_job(
    job_name="ulcerative-colitis-dysplasia-tile-masks-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=4,
    memory="48Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python scripts/run_tile_masks_pilot.py "
        f"--tiling-dir {TILING_DIR} --out {OUT_DIR} --workers 2",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
