#: Submit the 14-slide TILE-EMBEDDINGS job (prov-gigapath) for TCIA_HunCRC.
#:
#: Share-based twin of embeddings.py (the full-cohort submitter): the job's main
#: (scripts/embeddings_missing14.py) reads the tiling_missing14 parquets and the
#: whole slides straight from the mounts -- no MLflow URIs, no env vars. It embeds
#: only the 14 QC-gap slides and writes them into the EXISTING
#: preprocessing/embeddings/prov-gigapath/ dir as new {stem}.parquet files (the
#: 186 existing parquets are never touched; the script refuses to run if any of
#: the 14 stems is already there). After it, the dir holds all 200 and training
#: points at it unchanged.
#:
#: ONE job, not fourteen: the embed half is request-bound against the model
#: service (concurrency 512), so 14 parallel jobs would only make 14x the
#: requests to one service for no wall-clock gain. 389k tiles is ~10% of the
#: 3.9M-tile full-cohort job that finished in a morning.
#:
#: PREREQUISITE: the HUNCRC model service is up and serving prov-gigapath. The
#: script probes it with one white tile and exits in seconds against a dead
#: service (run scripts/embeddings_missing14.py --probe-only to check first).
#:
#: MOUNTS: DATA (crop_tiles -> read_slide_tiles opens the .mrxs whole slides) +
#: PROJECTS (tiling inputs and the embeddings output dir). Both load-bearing.
#:
#: SIZING: same as the full-cohort job -- the crop half is compute-bound (ray
#: tasks ask 4Gi/1cpu), the embed half request-bound.
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Defaults to configs/preprocessing/embeddings.yaml's svc (the huncrc rerolled
#: deployment); overridable for a re-deploy to a different RayService.
SVC = os.environ.get(
    "SVC",
    "http://rayservice-model-huncrc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
)

submit_job(
    job_name="ulcerative-colitis-dysplasia-embeddings-missing14-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="32Gi",
    shm="16Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run python scripts/embeddings_missing14.py --svc {SVC}",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
