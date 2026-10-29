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
#: MLflow works with public=True: the configmap URI is the in-cluster endpoint and
#: the access key exported below authenticates against it (proven by the provenance
#: registration jobs, Oct 29). The sensitive-cluster service name
#: http://mlflow.rationai-mlflow:5000 does NOT resolve on kuba-cluster -- do not use
#: it here (the export line that tried it is kept commented as the record of a dead
#: end).
#:
#: `job` is a dependency-GROUP here, not an extra -- uv add/uv run take `--group job`.
#: That group is only needed to RUN this submitter; the job itself syncs plain
#: `uv sync --frozen`, because the payload (`preprocessing.create_dataset`) never
#: imports kube_jobs -- only the submitters in scripts/ do.
#:   uv run --group job python scripts/preprocessing/create_dataset.py
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: public=True can only reach the OIDC-gated MLflow, and submit_job passes no password,
#: so the token has to be exported inside the job. Read from the notebook env rather than
#: written here: this file is tracked, and a literal would sit in git history forever.
#:   export MLFLOW_TRACKING_PASSWORD=<access key>   (UI -> Permission -> create access key)
#: Revoke the key after the run. `set +x` around the export keeps it out of the log.
#: Not secret; matches the author email in pyproject.toml and the account the key is under.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

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
        # username= does double duty (submit_job.py:110,190): the k8s label, which must
        # drop '@', and MLFLOW_TRACKING_USERNAME, which mlflow-oidc wants as the e-infra
        # email that the access key was minted for. The label-safe value above made the
        # token and the identity disagree -> 401. Override the MLflow side only.
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        # Provenance: capture_environment resolves the user from JOB_USER and fails the
        # run when it is unset or unregistered (kube-jobs 0.4.0 does not inject it).
        "export JOB_USER='borisim'",
        # BROKEN, kept as the record of a dead end: this is the sensitive cluster's
        # service name and does not resolve on kuba-cluster. See the MLflow note above.
        # "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        # --with overlays rationai-provenance (needed by the provenance hook) onto the
        # project env without touching pyproject/uv.lock. Simplify to a plain `uv run`
        # once the lib is a locked project dependency (uv add).
        "uv sync --frozen",
        "uv run --with 'rationai-provenance @ git+https://github.com/RationAI/provenance.git' "
        "python -m preprocessing.create_dataset +dataset=raw",
    ],
    # PROJECTS (read-write) added for provenance: verify_manifest reads
    # ${project_path}/preprocessing/manifest.csv written by the registration job.
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
