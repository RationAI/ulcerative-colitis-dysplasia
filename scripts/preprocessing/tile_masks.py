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
#: SIZING: memory is the level-N mask accumulator, one float32 array per column
#: sized to the whole slide extent, so it scales with SLIDE AREA. At level 2
#: (147200 x 136384 = 20 Gpx) that is ~80 GB per column: a single process_slide
#: reached 88.77GB and died at 91.36GB / 96.00GB even with --workers 1. The
#: pilot therefore renders at LEVEL 3 (68192 x 73600 = 5 Gpx, ~20 GB/column,
#: same level as the repo stage's config), so a 96Gi pod with --workers 1 has
#: room for the biggest slides plus the pyvips copy.
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
    memory="96Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python scripts/run_tile_masks_pilot.py "
        f"--tiling-dir {TILING_DIR} --out {OUT_DIR} --workers 1",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
