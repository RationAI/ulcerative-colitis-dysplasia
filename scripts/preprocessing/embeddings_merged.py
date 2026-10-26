#: Submit the full-cohort TILE-EMBEDDINGS job (prov-gigapath) for TCIA_HunCRC,
#: reading the MERGED local tiling dir -- no MLflow, no splits.
#:
#: Companion to scripts/preprocessing/embeddings.py (the stage-based submitter
#: that reads per-split tiling from MLflow URIs). That one needs a split
#: assignment and a tiling MLflow upload; neither exists yet. This one runs
#: scripts/run_embeddings_pilot.py against the flat output of
#: scripts/merge_tiling_shards.py -- the same code the pilot->stage pair
#: shares (crop_tiles / add_embedding_column / write_per_slide from
#: preprocessing/embeddings.py), same per-slide output layout
#: ({out}/{model}/{stem}.parquet), just all 186 slides in one output dir
#: because no folds are assigned yet (the tiling `fold` column is all None).
#:
#: PREREQUISITE: scripts/merge_tiling_shards.py has run, so
#: /mnt/projects/huncrc_dysplasia/tiling_186_merged/{slides,tiles} exists
#: (186 slides, 3,928,931 tiles).
#:
#: MOUNTS: DATA (crop_tiles -> read_slide_tiles opens every whole slide
#: itself) + PROJECTS (the merged tiling dir in, the per-slide parquets out).
#: No MLflow: the output lives on the share and is consumed locally; upload
#: it later if a split-aware stage needs it.
#:
#: SIZING: same crop arithmetic as the stage submitter -- crop tasks ask
#: 4Gi/1cpu, so cpu=8 bounds ~6-8 crops in flight within 32Gi, the rest is
#: driver + Ray object store (shm=16Gi). The 3.9 M-tile scale already
#: completed pod-locally at the same memory cap (embeddings_186, 2026-10-16).
#:
#: GATE: the script's first line is --probe -- one real tile cropped and
#: embedded end-to-end, so a scaled-to-zero prov-gigapath replica fails the
#: job in ~20 s with a clear message instead of dying mid-batch.
#:
#: RESUME: none -- the script materializes the full result and writes the
#: per-slide parquets at the end (a job restart redoes the run). The job pod
#: does not sleep or OOM at this size (same run completed pod-locally at the
#: same cap), so a restart is only an OOM tail risk, not an expected path.
#:
#: Needs nothing exported -- no MLflow in this job.
from kube_jobs import storage, submit_job


BRANCH = "huncrc"
TILING_DIR = "/mnt/projects/huncrc_dysplasia/tiling_186_merged"
OUT_DIR = "/mnt/projects/huncrc_dysplasia/embeddings_186_v3"
MODEL = "prov-gigapath"
#: huncrc rerolled deployment, same as configs/preprocessing/embeddings.yaml.
SVC = "http://rayservice-model-huncrc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000"

submit_job(
    job_name="ulcerative-colitis-dysplasia-embeddings-merged-huncrc",
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
        f"uv run python scripts/run_embeddings_pilot.py --probe --tiling-dir {TILING_DIR}",
        "uv run python scripts/run_embeddings_pilot.py "
        f"--tiling-dir {TILING_DIR} --out {OUT_DIR} "
        f"--model {MODEL} --concurrency 32",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
