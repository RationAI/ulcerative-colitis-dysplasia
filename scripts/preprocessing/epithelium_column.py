#: Submit the EPITHELIUM COLUMN job for TCIA_HunCRC.
#:
#: tiling_186_v3 ran before the epi-200 masks existed, so its `epithelium` column
#: is a stand-in copy of `tissue` (the run's epithelium_folder fell back to the
#: tissue masks). This job re-derives the real column WITHOUT re-tiling:
#: scripts/compute_epithelium_column.py loads each epi mask's level-2 top page
#: (147200x136384, the tiles' own resolution), binarizes at 0.5 probability
#: (the tissue_threshold: 0.5 convention), and gets every tile's epithelium
#: fraction in O(1) from an integral image. No whole-slide opens -- the part
#: that made tiling take ~3 h -- so the whole 186-slide job runs in ~30-60 min.
#:
#: Why a job instead of the notebook pod: same reason as the QC retry -- the
#: notebook can sleep or get OOM-killed mid-run, and this script is resumable
#: via per-slide part files (epithelium_column_parts/), so a dead pod costs at
#: most ~4 slides on re-run.
#:
#: MOUNTS: PROJECTS only. Everything read and written lives under
#: /mnt/projects/huncrc_dysplasia/ (tiling_186_v3, epithelium_masks/raw, the
#: parts dir, the output parquet) -- no slides are opened, so no DATA mount.
#: No MLflow either: the output is consumed locally by the merge step that
#: joins it onto the tiling tiles on (slide_id, x, y).
#:
#: SIZING: 4 workers x ~300 MB (uint8 level-2 page + float64 integral image
#: ~160 MB each) underestimated the peak -- the first submit (cpu=4/8Gi) was
#: OOMKilled (exit 137) after ~40 min, so 32Gi carries the merge/sort.
#:
#: Output: /mnt/projects/huncrc_dysplasia/epithelium_column_186.parquet with
#: columns (slide_id, x, y, epithelium) -- 3,928,931 rows, one per v3 tile.
#: The script asserts the epi top-page shape and that every tile fits inside
#: the mask extent, so a changed input fails loudly instead of writing wrong
#: fractions.
from kube_jobs import storage, submit_job


BRANCH = "huncrc"

submit_job(
    job_name="ulcerative-colitis-dysplasia-epithelium-column-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=4,
    memory="32Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python scripts/compute_epithelium_column.py",
    ],
    storage=[storage.public.PROJECTS],
)
