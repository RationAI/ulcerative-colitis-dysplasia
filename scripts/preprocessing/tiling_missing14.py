#: Submit one TILING job per missing hunCRC slide -- 14 jobs that run concurrently.
#:
#: WHY PER-SLIDE: the tiling pipeline's cost is a ~1.6M-tile full-slide mask-overlay
#: pass per slide, and Ray Data runs ~one task per pipeline stage per couple of CPUs.
#: One 8-CPU job for all 14 slides measures ~2 h; 14 jobs x cpu=2 tile every slide at
#: once -- ~20-30 min wall if the cluster has 28 CPU to spare, and the jobs simply
#: queue (still faster) if it does not. Each job also writes ONLY its own
#: {stem}.parquet files (scripts/tiling_missing14.py), so a single failed slide is a
#: single-job re-run and no two jobs can collide on a file.
#:
#: The job's main (scripts/tiling_missing14.py) reads every input from the share --
#: QC, tissue, epi, annotations, dataset_split.csv -- and needs no MLflow access, so
#: unlike the full-cohort submitter there are no artifact URIs to export. No env
#: vars at all.
#:
#: The 14 stems are dataset_split.csv minus preprocessing/tiling, computed
#: 2026-10-29. Re-derive (or pass a different --slides) if the split or the tiling
#: changes.
#:
#: MOUNTS: DATA (read_slides opens the .mrxs whole slides) + PROJECTS (mask inputs
#: and the preprocessing/tiling_missing14 output dir). Both load-bearing.
#:
#: The jobs clone BRANCH from GitHub -- scripts/tiling_missing14.py must be
#: committed and pushed before submitting.
from kube_jobs import storage, submit_job


BRANCH = "huncrc"

STEMS = [
    "011", "036", "039", "042", "045", "059", "064", "086",
    "091", "125", "155", "162", "174", "194",
]


for stem in STEMS:
    submit_job(
        job_name=f"ulcerative-colitis-dysplasia-tiling-{stem}-huncrc",
        username="borisim",
        image="cerit.io/rationai/base:2.0.6",
        public=True,
        cpu=2,
        #: Not 8Gi: Ray's OOM monitor reads the pod cgroup, and one slide's tile
        #: workers (~2.4 GB each) plus the .mrxs read cache already filled 8Gi --
        #: all 14 such jobs were OOMKilled at 7.63/8.00 GB on 2026-10-29.
        memory="32Gi",
        script=[
            f"git clone --branch {BRANCH} "
            "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
            "cd workdir",
            "uv sync --frozen",
            f"uv run python scripts/tiling_missing14.py --slides {stem}",
        ],
        storage=[storage.public.DATA, storage.public.PROJECTS],
    )
