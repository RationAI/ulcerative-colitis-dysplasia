#: Submit annotation-mask creation for TCIA_HunCRC as a cluster job.
#:
#: Same hunCRC setup as scripts/preprocessing/quality_control.py -- public cloud,
#: BRANCH pinned, MLflow token exported into the job, +dataset=raw.
#:
#: Storage needs BOTH mounts, which is the one place this stage differs from every
#: other preprocessing job here:
#:   * public.PROJECTS -- reads the EMPIA JSONs out of ${project_path}/annotations and
#:     writes ${project_path}/annotation_masks. The only read-write mount.
#:   * public.DATA -- opens each .mrxs for slide_resolution, so the canvas and mpp come
#:     from the slide rather than a constant. QC needs neither: it POSTs slide_path
#:     strings to the QC service and never opens a slide itself
#:     (preprocessing/quality_control.py:95-105). tissue_masks needs DATA but not
#:     PROJECTS as an input.
#:
#: +dataset=raw rather than +dataset=processed. configs/preprocessing.yaml has
#: `dataset: ???` so a dataset group must be named, but this stage reads no CSV -- it
#: globs the annotation dir. processed.yaml also still carries the IBD cohort's
#: artifact URI (configs/dataset/processed.yaml:6), which autolog would record on the
#: run and nobody would read.
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
#:
#: The job clones configs/ and preprocessing/ from GitHub at BRANCH below, so a local
#: edit to annotation_masks.yaml or the payload has no effect until it is pushed.
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as create_dataset.py -- see the notes there for why the k8s label and the
#: MLflow identity differ, and why the token comes from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

#: 200 slides x 2 classes at level 5 (313 MB canvas each). The ray task is decorated
#: memory=2Gi and process_items runs max_concurrent=4 of them, so 8 GiB of the 16 GiB
#: is the tasks; the rest is pyvips and tifffile scratch. cpu=8 matches the other mask
#: submitters; the work is one slide read plus one rasterize per class, so it is
#: memory-bound rather than core-bound.
submit_job(
    job_name="ulcerative-colitis-dysplasia-annotation-masks-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=8,
    memory="16Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        "uv run python -m preprocessing.annotation_masks +dataset=raw",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
