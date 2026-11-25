#: Submit the full-cohort TILE-EMBEDDINGS job (prov-gigapath) for TCIA_HunCRC.
#:
#: Same hunCRC setup as tiling.py / epithelium_masks.py -- public cloud, BRANCH pinned,
#: MLflow token exported into the job. Runs preprocessing/embeddings.py (ported from the
#: IBD feature/embeddings branch), which reads each split's tiling parquet, crops every
#: tile to uint8 RGB and POSTs it to the embedding model, writing one parquet per slide
#: ({project_path}/embeddings/{model}/{stem}.parquet) -- the layout the training
#: consumer (feat/modeling DysplasiaEmbeddingsDataset) globs by slide stem.
#:
#: "tiling_filtered" vs "tiling": the ported stage reads
#: config.dataset.mlflow_uris.tiling_filtered (the IBD chain's name), but it only needs
#: each split's `slides` + `tiles` folders -- which the huncrc tiling job uploads
#: UNFILTERED to <tiling-run>/artifacts/tiling/<split>. "All slides" embedding does not
#: need the IBD filter_tiles stage at all, so the three TILING_* URIs below point at the
#: tiling job's artifacts directly.
#:
#: PREREQUISITES, in order: (1) tiling finished for all 200 and you have its run id;
#: (2) the HUNCRC model service is up and serving prov-gigapath (the stage pre-flight
#: GETs {svc}/openapi.json and refuses to start against a dead service). The service is
#: the same rerolled rayservice-model-huncrc deployment the episeg job uses -- one
#: reroll serves both.
#:
#: MOUNTS: DATA (crop_tiles -> read_slide_tiles opens every whole slide itself) +
#: PROJECTS (output_dir ${project_path}/embeddings is pod-share, uploaded at the end).
#:
#: SIZING: the crop half is compute-bound (ray tasks ask 4Gi/1cpu,
#: preprocessing/embeddings.py crop_tiles), the embed half is request-bound
#: (concurrency 512 in configs/preprocessing/embeddings.yaml). cpu=8/32Gi: 8 cpus bound
#: the 4Gi crops to ~6 in flight (~24Gi), the rest is driver + actor pool; shm=16Gi is
#: ray's object store.
#:
#: OUTPUT SURVIVAL, same shape as the tissue job: per-slide parquets accumulate under
#: /mnt/projects and are logged once at the end -- a crash near the END loses the
#: upload but the files survive on the share (re-run the stage; write_per_slide
#: overwrites the same {stem}.parquet). The embedding job is the long pole of the
#: morning: budget many hours.
#:
#: Needs exported in the shell that runs this (all four TILING_*; no safe defaults --
#: the configs still carry IBD experiment-111 URIs that would embed the wrong cohort):
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
#:   export TILING_TRAIN_URI=mlflow-artifacts:/168/<tiling-run>/artifacts/tiling/train
#:   export TILING_TEST_PRELIMINARY_URI=mlflow-artifacts:/168/<tiling-run>/artifacts/tiling/test_preliminary
#:   export TILING_TEST_FINAL_URI=mlflow-artifacts:/168/<tiling-run>/artifacts/tiling/test_final
#:   export SVC=...   (optional; defaults to the huncrc model service in the config)
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as the other huncrc submitters -- see quality_control.py for why the k8s
#: label and the MLflow identity differ, and why the token comes from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
#: Defaults to configs/preprocessing/embeddings.yaml's svc (the huncrc rerolled
#: deployment); overridable for a later re-deploy to a different RayService.
SVC = os.environ.get(
    "SVC",
    "http://rayservice-model-huncrc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
)

TILING_TRAIN_URI = os.environ.get("TILING_TRAIN_URI")
TILING_TEST_PRELIMINARY_URI = os.environ.get("TILING_TEST_PRELIMINARY_URI")
TILING_TEST_FINAL_URI = os.environ.get("TILING_TEST_FINAL_URI")


def _require(name: str, value: str | None) -> str:
    if not value:
        raise SystemExit(
            f"{name} is unset: export the tiling job's run artifact "
            f"mlflow-artifacts:/168/<tiling-run>/artifacts/tiling/<split>."
        )
    return value


if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )
TILING_TRAIN_URI = _require("TILING_TRAIN_URI", TILING_TRAIN_URI)
TILING_TEST_PRELIMINARY_URI = _require(
    "TILING_TEST_PRELIMINARY_URI", TILING_TEST_PRELIMINARY_URI
)
TILING_TEST_FINAL_URI = _require("TILING_TEST_FINAL_URI", TILING_TEST_FINAL_URI)

submit_job(
    job_name="ulcerative-colitis-dysplasia-embeddings-huncrc",
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
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        "uv run python -m preprocessing.embeddings +dataset=processed_w_masks "
        f"++dataset.mlflow_uris.tiling_filtered.train={TILING_TRAIN_URI} "
        f"++dataset.mlflow_uris.tiling_filtered.test_preliminary="
        f"{TILING_TEST_PRELIMINARY_URI} "
        f"++dataset.mlflow_uris.tiling_filtered.test_final={TILING_TEST_FINAL_URI} "
        f"svc={SVC} "
        "project_path=/mnt/projects/huncrc_dysplasia",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
