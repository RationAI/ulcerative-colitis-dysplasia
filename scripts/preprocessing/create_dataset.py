#: Submit create_dataset for TCIA_HunCRC as a cluster job.
#:
#: Two things here differ from the IBD submitters in this directory, and both are
#: because HunCRC lives on the PUBLIC cloud, not the sensitive one:
#:
#:   * public=True -- selects the Public Cloud cluster.
#:   * storage.public.DATA -- PVC rationai-data-ro-pvc-jobs (read-only rationai/Data),
#:     mounted at /mnt/data/Public. Hence the doubled "Public" in configs/dataset/raw.yaml:
#:     /mnt/data/Public + /Public/colorectum/TCIA_HunCRC.
#:     secure.DATA would also expose that mount (it carries both data-ro and this PVC) but
#:     additionally mounts the sensitive MOU store at /mnt/data, and its `data-ro` PVC does
#:     not exist on the public cluster -- so with public=True this is the only valid choice.
#:
#: MLflow does not work with public=True, which is the open problem here. Only two
#: deployments exist and each is blocked one way (job log, run 1, Oct 1):
#:
#:   * the configmap URI submit_job injects is the OIDC-gated public instance, and it
#:     passes only MLFLOW_TRACKING_USERNAME -- no password param -- so mlflow gets the
#:     login HTML where it expects JSON.
#:   * http://mlflow.rationai-mlflow:5000 is the SENSITIVE cluster's service name; on
#:     kuba-cluster it fails DNS ("Failed to resolve 'mlflow.rationai-mlflow'"). It is in
#:     NO_PROXY only because kube_jobs was written for the sensitive side.
#:
#: Disabling logging is not an escape: the payload writes dataset.csv to a tmpdir and
#: MLflow is its only durable sink (preprocessing/create_dataset.py:124-126).
#:
#: `job` is a dependency-GROUP here, not an extra -- uv add/uv run take `--group job`.
#: That group is only needed to RUN this submitter; the job itself syncs plain
#: `uv sync --frozen`, because the payload (`preprocessing.create_dataset`) never
#: imports kube_jobs -- only the submitters in scripts/ do.
#:   uv run --group job python scripts/preprocessing/create_dataset.py
from kube_jobs import storage, submit_job


BRANCH = "huncrc"

submit_job(
    job_name="ulcerative-colitis-dysplasia-create-dataset-huncrc",
    # Flows verbatim into labels={"created_by": username}; k8s labels forbid '@', so the
    # author email from pyproject.toml is rejected with 422. GitLab username instead.
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=2,
    memory="8Gi",
    script=[
        # The clone must pin BRANCH: the default branch is master, which still has the
        # *.czi version of create_dataset.
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        # BROKEN, kept as the record of a dead end: this is the sensitive cluster's
        # service name and does not resolve on kuba-cluster. See the MLflow note above.
        # "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv sync --frozen",
        "uv run python -m preprocessing.create_dataset +dataset=raw",
    ],
    storage=[storage.public.DATA],
)
