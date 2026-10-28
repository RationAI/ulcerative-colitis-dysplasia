#: Submit the 14-slide QC RETRY for TCIA_HunCRC as a cluster job.
#:
#: Same hunCRC setup as quality_control.py / tissue_masks.py -- public cloud, BRANCH
#: pinned, MLflow token exported into the job. Runs scripts/rerun_missing_qc.py, which
#: recomputes the missing set live (slides dir minus qc_metrics.csv rows) instead of
#: reprocessing all 200 (~6 days), waits ~10 h at max_concurrent 5, organizes the new
#: masks into the 4 subdirs, merges qc_metrics.csv to 200 rows, and -- with --upload --
#: logs the finished tree to a fresh MLflow run as `qc_output` (preprocessing/tiling.py:338
#: downloads that artifact; the /mnt/projects tree alone is not consumed by tiling).
#:
#: MOUNTS: DATA (the script LISTS the slides dir to compute the missing set and check
#: paths -- the service itself opens the slides) + PROJECTS (the job dir is on the
#: rationai-projects share, where the 186 good masks live and the 14 new ones land).
#:
#: SIZING: the pod only holds HTTP connections while the QC service does the work, so
#: cpu=4/8Gi is plenty (the original job's 8/16Gi is for the same request-bound stage).
#:
#: The pod's environment carries the in-cluster MLflow tracking URI
#: (http://mlflow.rationai-mlflow:5000/), which is why the --upload step works in-job
#: even though a notebook needs the public URI (memory: mlflow-access-topology).
#:
#: GATE THE SERVICE BEFORE SUBMITTING: the script probes /openapi.json (120 s x3) and
#: refuses to run against a dead service, but that only distinguishes "cold replica
#: still spawning" from "down". As of the last check the QC service had not answered
#: any probe since 2026-07-21 -- if it needs the same reroll as the model service, do
#: that FIRST or this job spends its gate then exits.
#:
#: The printout at the end is the handoff to the tiling job:
#:   export QC_ARTIFACT_URI='<printed mlflow-artifacts:/168/<run>/artifacts/qc_output>'
#:
#: Needs exported in the shell that runs this:
#:   export MLFLOW_TRACKING_PASSWORD=<access key>
import os

from kube_jobs import storage, submit_job


BRANCH = "huncrc"

#: Same values as quality_control.py -- see the notes there for why the k8s label and
#: the MLflow identity differ, and why the token comes from the environment.
MLFLOW_USER = "519192@mail.muni.cz"
MLFLOW_TOKEN = os.environ.get("MLFLOW_TRACKING_PASSWORD")
#: The original job's output dir: 186 done slides + 4 mask subdirs + qc_metrics.csv.
#: The script merges INTO it (safe merge, per-slide csvs only) and uploads the whole
#: tree, so the artifact looks exactly like the original job's `qc_output` would.
QC_OUTPUT_DIR = "/mnt/projects/huncrc_dysplasia/quality_control"

if not MLFLOW_TOKEN:
    raise SystemExit(
        "MLFLOW_TRACKING_PASSWORD is unset. Mint an access key in the MLflow UI "
        "(Permission -> create access key) and export it. A login password is rejected."
    )

submit_job(
    job_name="ulcerative-colitis-dysplasia-qc-retry-huncrc",
    username="borisim",
    image="cerit.io/rationai/base:2.0.6",
    public=True,
    cpu=4,
    memory="8Gi",
    script=[
        f"git clone --branch {BRANCH} "
        "https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        f"set +x; export MLFLOW_TRACKING_USERNAME='{MLFLOW_USER}'; "
        f"export MLFLOW_TRACKING_PASSWORD='{MLFLOW_TOKEN}'; set -x",
        "uv sync --frozen",
        #: 059 is the one slide the service needs longer than the 5 h default for: it
        #: timed out at exactly 18000 s twice (once alone), leaving its per-tile masks
        #: but no qc_metrics.csv row, so --slides pins the retry to it and the timeout
        #: doubles. The other 199 rows are already in the csv.
        "uv run python scripts/rerun_missing_qc.py "
        f"--out {QC_OUTPUT_DIR} --slides 059 --timeout 36000 "
        "--max-concurrent 1 --upload",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
