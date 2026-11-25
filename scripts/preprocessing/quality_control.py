#: Submit quality_control for TCIA_HunCRC as a cluster job.
#:
#: Same hunCRC setup as scripts/preprocessing/split_dataset.py -- public cloud, BRANCH
#: pinned, MLflow token exported into the job, +dataset=processed with the artifact URI
#: overridden to the hunCRC run.
#:
#: QC differs from both earlier stages in where its data comes from: it opens no slide
#: itself. It reads slide_path STRINGS out of dataset.csv (preprocessing/quality_control.py
#: :95-96, :105) and POSTs each one to the RationAI QC service (.venv/.../rationai/
#: resources/qc.py:50, "wsi_path": str(wsi_path)), which fetches the slide from wherever
#: IT can see it. So the pod needs no data mount -- only a writable staging dir for the
#: returned masks, which then go to MLflow as `qc_output` (:112). Tiling later reads that
#: as dataset.mlflow_uris.qc (preprocessing/tiling.py:303).
#:
#: output_dir is overridden on the command line even though configs/base.yaml now points
#: project_path at /mnt/projects/huncrc_dysplasia (so qc.yaml:3's ${project_path}/
#: quality_control already resolves to the same place). It stays explicit because the job
#: clones configs from GitHub at BRANCH: an unpushed base.yaml would silently give the job
#: the IBD cohort's staging dir, and a wrong-but-writable path fails noisily at best.
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
#:   export DATASET_ARTIFACT_URI=...   (optional; defaults to the hunCRC create_dataset run)
#:   export QC_OUTPUT_DIR=...          (optional; defaults to the path below)
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as create_dataset.py -- see the notes there for why the k8s label and the
#: MLflow identity differ, and why the token comes from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
DATASET_ARTIFACT_URI = os.environ.get(
    "DATASET_ARTIFACT_URI",
    "mlflow-artifacts:/168/2ca0c1928c2c48acb56a5110efefb957/artifacts/dataset.csv",
)
#: Must live under /mnt/projects, which is what storage.public.PROJECTS mounts read-write
#: (PVC rationai-projects-pvc-jobs). Anything else the pod writes is lost when it exits.
QC_OUTPUT_DIR = os.environ.get(
    "QC_OUTPUT_DIR", "/mnt/projects/huncrc_dysplasia/quality_control"
)

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-quality-control-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    # QC is request-bound, not compute-bound: max_concurrent 5 slide checks over HTTP
    # (configs/preprocessing/qc.yaml:5). The IBD submitter's cpu=8/16Gi is for the same
    # stage, kept here because per-pixel tiff masks for ~200 slides accumulate in the
    # staging dir before upload.
    cpu=8,
    memory="16Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        "uv run python -m preprocessing.quality_control +dataset=processed "
        f"++dataset.mlflow_uris.dataset={DATASET_ARTIFACT_URI} "
        f"output_dir={QC_OUTPUT_DIR}",
    ],
    storage=[storage.public.PROJECTS],
)
