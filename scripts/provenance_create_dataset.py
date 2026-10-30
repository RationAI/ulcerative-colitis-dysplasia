#: Submit the provenance cut-down of the dataset-creation step as a cluster job.
#:
#: Runs scripts/provenance/create_dataset.py — a SEPARATE cut-down that reuses
#: the original preprocessing.create_dataset logic but leaves the original
#: script untouched (production reruns stay clean). Verifies the 200-slide
#: registered cohort, lists the dataset on a SUBSET_N sample, and logs the
#: run with provenance (env, config, output manifest, pinned input run) into
#: the isolated "provenance_huncrc_test" experiment.
#:
#: Cheap: no imaging, no tiling — glob + regex + MLflow logging only.
#:
#: Storage: DATA (read the .mrxs dir listing) + PROJECTS (read the registration
#: manifest). The job writes nothing to the project share.
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
import os

from kube_jobs import storage, submit_job

BRANCH = "huncrc"
#: The Rancher username — k8s label + provenance identity (exported as JOB_USER
#: into the job; the provenance lib resolves the user from it and fails if unset).
USERNAME = "borisim"

#: Same values as the other huncrc submitters -- see embeddings_huncrc.py.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-prov-create-dataset",
    username=USERNAME,
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=2,
    memory="8Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        # Provenance: capture_environment resolves the user from JOB_USER and
        # fails the run when it is unset or unregistered (kube-jobs 0.4.0 does
        # not inject it).
        f"export JOB_USER='{USERNAME}'",
        "uv sync --frozen",
        # --with overlays rationai-provenance onto the project env without
        # touching pyproject/uv.lock (the production lock stays pristine).
        "uv run --with 'rationai-provenance @ git+https://github.com/RationAI/provenance.git' "
        "python -m scripts.provenance.create_dataset",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
