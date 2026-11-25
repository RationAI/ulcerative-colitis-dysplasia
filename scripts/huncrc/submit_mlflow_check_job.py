"""Submit Step 10, stage 1 as a Kubernetes job: MLflow read-only check + dry run.

The Jupyter pod cannot authenticate to the MLflow server; jobs get its URI from
the ``default-envs`` configmap and ``MLFLOW_TRACKING_USERNAME`` from kube_jobs.
This job clones the branch, runs ``uv sync --frozen`` and then

  1. ``log_to_mlflow.py --check-mlflow``: experiment 168 name and artifact
     location, and whether the job's credentials are enough to read it;
  2. ``log_to_mlflow.py --dry-run`` for the six final model sets.

It then exits with the check's status. It writes nothing to MLflow: the pod
script has no logging call at all. Output goes to ``<out>/job.log`` and the dry
run to ``<out>/dryrun.txt``. Credentials are never printed, only whether set.

    python scripts/huncrc/submit_mlflow_check_job.py --dry-run   # print, submit nothing
    python scripts/huncrc/submit_mlflow_check_job.py             # submit
"""

from __future__ import annotations

import argparse
import shlex


PUBLIC = True  # kuba-cluster, as for the Step-9 job

REPO_URL = "https://github.com/RationAI/ulcerative-colitis-dysplasia.git"
BRANCH = "feature/huncrc-tile-classifier"
TC = "/mnt/projects/huncrc_dysplasia/tile_classifier"
REPORT = f"{TC}/step9_mlp/report.json"
DEFAULT_OUT = f"{TC}/step10_mlflow"
# The report records the Jupyter paths of A and B/C; their small files were
# copied to the share with the same layout.
SOURCE_MAP = [
    f"/home/jovyan/huncrc_tile_classifier_results/2026-11-21={TC}",
    f"/home/jovyan/huncrc_tile_classifier_results={TC}",
]

CPU = 2
MEMORY = "4Gi"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--job-name", default="ulcerative-colitis-dysplasia-mlflow-check",
                   help="kube_jobs appends -<username>")  # fmt: skip
    p.add_argument("--branch", default=BRANCH)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true", help="print, do not submit")
    a = p.parse_args(argv)
    if not a.out.startswith("/mnt/projects/"):
        p.error("--out must be under /mnt/projects/")
    return a


def pod_script(a: argparse.Namespace) -> list[str]:
    q = shlex.quote
    logger = "uv run --frozen --no-sync python scripts/huncrc/log_to_mlflow.py"
    dry = (
        f"{logger} --dry-run --report {q(REPORT)} --source-map "
        + " ".join(q(m) for m in SOURCE_MAP)
        + f" --dry-run-out {q(a.out + '/dryrun.txt')}"
    )
    body = "; ".join(
        [
            "set -uo pipefail",  # no -x: never trace the environment
            "echo \"[job] $(date '+%F %T') host=$(hostname)\"",
            f"git clone --depth 1 --single-branch -b {q(a.branch)} {REPO_URL} workdir"
            " || exit 9",
            "cd workdir || exit 9",
            "git log -1 --format='[job] commit %H %s'",
            "uv sync --frozen || exit 10",
            "echo '[job] stage 1: read-only MLflow check'",
            f"{logger} --check-mlflow; rc=$?",
            'echo "[job] check exit code $rc"',
            "echo '[job] stage 2: dry run (nothing is sent to MLflow)'",
            f"{dry} > /dev/null || exit 11",
            f"echo '[job] dry run written to {a.out}/dryrun.txt'",
            "echo \"[job] $(date '+%F %T') finished\"",
            'exit "$rc"',
        ]
    )
    log = q(f"{a.out}/job.log")
    return [
        f"mkdir -p {q(a.out)}",
        f"( {body} ) 2>&1 | tee -a {log}",
        'exit "${PIPESTATUS[0]}"',
    ]


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    script = pod_script(a)
    print(f"job {a.job_name}: public={PUBLIC} cpu={CPU} memory={MEMORY}")
    print("\n".join(script))
    if a.dry_run:
        return

    import inspect

    from kube_jobs import storage, submit_job

    if "username" in inspect.signature(submit_job).parameters:
        raise SystemExit(
            "this kube_jobs predates 7c2a73c and needs username=; use the installed "
            "/opt/venv one (python on PATH) or upgrade the pin"
        )
    mounts = storage.public if PUBLIC else storage.secure
    submit_job(
        job_name=a.job_name,
        public=PUBLIC,
        cpu=CPU,
        memory=MEMORY,
        script=script,
        storage=[mounts.PROJECTS],  # report and runs are all under /mnt/projects
    )


if __name__ == "__main__":
    main()
