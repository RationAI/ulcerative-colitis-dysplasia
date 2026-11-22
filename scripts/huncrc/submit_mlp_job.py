"""Submit Step 9 (MLP heads D/E, within-slide negatives, 5-fold CV) as a Kubernetes job.

Same pattern as ``scripts/preprocessing/tiling.py`` (master) and
``scripts/preprocessing/embeddings.py`` (feature/embeddings): the pod clones this
repo at ``--branch`` (default ``feature/huncrc-tile-classifier``), runs
``uv sync --frozen`` and then ``scripts/huncrc/run_cv.sh train`` followed by
``report``. Everything the pod prints, ``set -x`` traces included, goes to
``<out>/run.log``. ``report.json`` and ``table.txt`` are written to ``<out>``
and the per-(variant, fold) runs to ``<out>/runs``.

The pod clones the branch from GitHub, so the code must be pushed first. The
feature caches and the results live on the mounted project share (never in the
pod or in /home/jovyan). A missing cache is rebuilt from the embeddings. An
interrupted job resumes when it is resubmitted with the same ``--out``.
test_final is never read and nothing goes to MLflow.

    # print the pod script, submit nothing (does not need kube_jobs)
    python scripts/huncrc/submit_mlp_job.py --smoke --dry-run
    # submit (needs the kube-jobs package, see the "job" dependency group)
    python scripts/huncrc/submit_mlp_job.py --smoke     # fold 0, 1 epoch, 300 tiles/slide
    python scripts/huncrc/submit_mlp_job.py             # full: D E x 5 folds x 10 epochs
"""

from __future__ import annotations

import argparse
import shlex


USERNAME = "..."  # fill in: your RationAI username (job label "created_by")
PUBLIC = False  # False = secure cluster (kubas-cluster), True = kuba-cluster

REPO_URL = "https://github.com/RationAI/ulcerative-colitis-dysplasia.git"
BRANCH = "feature/huncrc-tile-classifier"
SHARE = "/mnt/projects/huncrc_dysplasia"
PRE = f"{SHARE}/preprocessing"
SLIDE_CSV = "/mnt/data/Public/Public/colorectum/TCIA_HunCRC/slide_level_annotations.csv"
DEFAULT_CACHE = f"{SHARE}/tile_classifier/cache"
DEFAULT_OUT = f"{SHARE}/tile_classifier/step9_mlp"
DEFAULT_SMOKE_OUT = f"{SHARE}/tile_classifier/step9_mlp_smoke"
PERSISTENT_PREFIX = "/mnt/projects/"  # mounted read-write in the pod

CPU = 8
MEMORY = "32Gi"
SHM = "16Gi"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--username", default=USERNAME)
    p.add_argument("--job-name", help="default: ...-huncrc-mlp[-smoke]")
    p.add_argument("--branch", default=BRANCH)
    p.add_argument(
        "--public",
        action=argparse.BooleanOptionalAction,
        default=PUBLIC,
        help="cluster, which also selects the matching kube_jobs storage set",
    )
    p.add_argument(
        "--cache-root", default=DEFAULT_CACHE, help="feature caches (reused/rebuilt)"
    )
    p.add_argument(
        "--out", help=f"results dir (default {DEFAULT_OUT}, or {DEFAULT_SMOKE_OUT})"
    )
    p.add_argument("--variants", nargs="+", default=["D", "E"])
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--neg-cap", type=int, default=5000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument(
        "--a-runs", default="", help="report: Step-4a runs dir on the share (optional)"
    )
    p.add_argument(
        "--other-runs",
        nargs="*",
        default=[],
        help="report: Step-8 B/C runs dir(s) on the share (optional)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="plumbing test: fold 0, 1 epoch, 300 tiles per slide, 20 bootstraps",
    )
    p.add_argument("--dry-run", action="store_true", help="print, do not submit")
    a = p.parse_args(argv)
    if a.smoke:
        a.folds, a.epochs, a.n_boot, a.smoke_tiles = [0], 1, 20, 300
    else:
        a.smoke_tiles = None
    a.out = a.out or (DEFAULT_SMOKE_OUT if a.smoke else DEFAULT_OUT)
    a.job_name = a.job_name or (
        "ulcerative-colitis-dysplasia-huncrc-mlp" + ("-smoke" if a.smoke else "")
    )
    for name in ("cache_root", "out"):
        if not getattr(a, name).startswith(PERSISTENT_PREFIX):
            p.error(f"--{name.replace('_', '-')} must be under {PERSISTENT_PREFIX}")
    for d in [a.a_runs, *a.other_runs]:
        if d and not d.startswith("/mnt/"):
            p.error(f"{d}: runs dirs must be on mounted storage (/mnt/...)")
    if not set(a.variants) <= {"B", "C", "D", "E"}:
        p.error("--variants must be among B C D E")
    if not a.dry_run and a.username in ("", "..."):
        p.error("set USERNAME in this file or pass --username")
    return a


def pod_script(a: argparse.Namespace) -> list[str]:
    """Shell lines; kube_jobs joins them with ';' after its git-credential setup."""
    q = shlex.quote
    env = {
        "PRE": PRE,
        "SLIDE_CSV": SLIDE_CSV,
        "CACHE_ROOT": a.cache_root,
        "OUT": a.out,
        "A_RUNS": a.a_runs,
        "OTHER_RUNS": " ".join(a.other_runs),
        "VARIANTS": " ".join(a.variants),
        "FOLDS": " ".join(map(str, a.folds)),
        "EPOCHS": str(a.epochs),
        "NEG_CAP": str(a.neg_cap),
        "N_BOOT": str(a.n_boot),
        "SMOKE_TILES": "" if a.smoke_tiles is None else str(a.smoke_tiles),
    }
    exports = " ".join(f"{k}={q(v)}" for k, v in env.items())
    body = "; ".join(
        [
            "set -euo pipefail",
            "echo \"[job] $(date '+%F %T') host=$(hostname) nproc=$(nproc) "
            "cpu.max=$(cat /sys/fs/cgroup/cpu.max) "
            'memory.max=$(cat /sys/fs/cgroup/memory.max)"',
            f"git clone --depth 1 --single-branch -b {q(a.branch)} {REPO_URL} workdir",
            "cd workdir",
            "git log -1 --format='[job] commit %H %s'",
            "uv sync --frozen",
            f"export {exports}",
            "bash scripts/huncrc/run_cv.sh train",
            "bash scripts/huncrc/run_cv.sh report",
            "echo \"[job] $(date '+%F %T') finished\"",
        ]
    )
    log = q(f"{a.out}/run.log")
    return [
        f"mkdir -p {q(a.out)} {q(a.cache_root)}",
        f"( {body} ) 2>&1 | tee -a {log}",
        'exit "${PIPESTATUS[0]}"',
    ]


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    script = pod_script(a)
    print(f"job {a.job_name}: public={a.public} cpu={CPU} memory={MEMORY} shm={SHM}")
    print("\n".join(script))
    if a.dry_run:
        return

    from kube_jobs import storage, submit_job

    # DATA mounts /mnt/data/Public (read-only; the HunCRC slide-level CSV),
    # PROJECTS mounts /mnt/projects (read-write; embeddings, tiling, annotations,
    # caches and results). The secure and public sets mount the same paths.
    mounts = storage.public if a.public else storage.secure
    submit_job(
        job_name=a.job_name,
        username=a.username,
        public=a.public,
        cpu=CPU,
        memory=MEMORY,
        shm=SHM,
        script=script,
        storage=[mounts.DATA, mounts.PROJECTS],
    )


if __name__ == "__main__":
    main()
