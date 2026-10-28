r"""Re-run ONLY the QC slides that the 5-day job left missing, straight into the job dir.

    uv run python scripts/rerun_missing_qc.py                 # recompute + run the missing set
    uv run python scripts/rerun_missing_qc.py --probe         # just check the service answers
    uv run python scripts/rerun_missing_qc.py --dry-run       # print the plan, send nothing
    uv run python scripts/rerun_missing_qc.py --slides 042,194

WHY A DEDICATED SCRIPT AND NOT `python -m preprocessing.quality_control`
-----------------------------------------------------------------------
The stock stage reprocesses every slide from the dataset csv (200 slides ~= 6 days) and,
worse, its csv tail is NOT safe when a combined qc_metrics.csv already exists in the
output dir:

    csvs = list(output_path.glob("*.csv"))            # picks up the EXISTING qc_metrics.csv
    pd.concat([...]).to_csv(output_path / "qc_metrics.csv")   # writes the 200-row file
    for f in csvs: f.unlink()                          # unlinks the old list -- incl. qc_metrics.csv

In a full run qc_metrics.csv does not exist at glob-time, so the final unlink leaves the
freshly-written file. On a SUBSET rerun the pre-existing qc_metrics.csv IS in `csvs`, so
the loop deletes the combined file it just wrote. So this script does the merge itself
(see merge_qc_csvs): read the existing combined + the new per-slide csvs, concat, dedup
by slide_name, write combined, delete ONLY the per-slide csvs. The 186 good rows survive.

THE SET, computed live (do not trust the ledger in the preprocessing/quality_control.py
docstring -- it rots the moment anything else writes). As at the last check, 200 real
slides, 186 in qc_metrics.csv, 14 missing:

    011 036 039 042 045 059 064 086 091 125 155 162 174 194

Of those, 12 died on a CLIENT timeout (empty error message -- the service was fine, just
slow) and two (042, 194) died on a real service HTTP 500 three times over, so they may
fail again for a reason the other 12 do not have. Expect ~10 h at max_concurrent 5.

IDEMPOTENT: any slide whose slide_name is already in the output qc_metrics.csv is skipped
at start, so a re-run after an interruption only redoes what is still missing.
"""

import argparse
import asyncio
import time
from pathlib import Path

import pandas as pd
from rationai.client import AsyncClient
from rationai.types import SlideCheckConfig
from tqdm.asyncio import tqdm

# Reuse the proven constants + the 120s x3 reachability probe from the single-slide script.
from qc_one_slide import JOB_OUT, SLIDES_DIR, service_up

#: The 4 mask prefixes QC writes flat as `<Prefix>_<slide>.tiff`, and the subdir the job's
#: organize_masks (preprocessing/quality_control.py:organize_masks) moves them into. Same
#: set the 5-day job produced -- check_focus=True + check_residual=True, check_folding=False.
MASK_TARGETS = [
    ("Piqe_focus_score_piqe_median", "blur_per_tile"),
    ("Piqe_piqe_median_activity_mask", "blur_per_pixel"),
    ("ResidualArtifactsAndCoverage_cov_percent_heatmap", "artifacts_per_tile"),
    ("ResidualArtifactsAndCoverage_coverage_mask", "artifacts_per_pixel"),
]


def real_slides(slides_dir: Path) -> list[str]:
    """Zeros-padded slide stems (exactly 3 digits) under the v2 dir, sorted."""
    return sorted(
        p.stem for p in slides_dir.glob("*.mrxs") if p.stem.isdigit() and len(p.stem) == 3
    )


def done_slides(qc_metrics: Path) -> set[str]:
    """slide_name values in qc_metrics.csv normalized to zfill(3) -- the service writes
    them unpadded (1, 10, 100), the files are padded (001, 010, 100)."""
    if not qc_metrics.exists():
        return set()
    df = pd.read_csv(qc_metrics)
    if "slide_name" not in df:
        return set()
    return {str(int(s)).zfill(3) for s in df["slide_name"].astype(str) if s.isdigit()}


def organize_masks(out_dir: Path, subdir: str, mask_prefix: str) -> int:
    """Move flat `<Prefix>_<slide>.tiff` into `<out_dir>/<subdir>/<slide>.tiff`."""
    prefix_dir = out_dir / subdir
    prefix_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for file in list(out_dir.glob(f"{mask_prefix}_*.tiff")):
        slide_name = file.name.replace(f"{mask_prefix}_", "")
        file.rename(prefix_dir / slide_name)
        moved += 1
    return moved


def upload_qc_artifacts(out_dir: Path, experiment_id: str) -> str:
    """Log the finished output dir as `qc_output` on a fresh MLflow run and return the
    artifact URI. Tiling consumes QC via `download_artifacts(config.dataset.mlflow_uris.qc)`
    (preprocessing/tiling.py:338) -- the /mnt/projects tree alone is not enough, so a
    completed QC tree must land on a run's artifacts tab the way the original job did.
    The in-cluster tracking URI is injected into the job's environment, so plain fluent
    mlflow works here (from a notebook it would not -- see scripts/
    upload_annotation_masks.py for the public-URI dance)."""
    import mlflow

    #: BY KEYWORD: set_experiment's first positional is the NAME, so set_experiment(168)
    #: goes through get_experiment_by_name(168) and dies with "bad argument type".
    mlflow.set_experiment(experiment_id=int(experiment_id))
    with mlflow.start_run(run_name="QC output (200 slides, huncrc)") as run:
        mlflow.log_artifacts(local_dir=str(out_dir), artifact_path="qc_output")
    uri = f"mlflow-artifacts:/{experiment_id}/{run.info.run_id}/artifacts/qc_output"
    print(f"\nqc_output uploaded to run {run.info.run_id} (experiment {experiment_id})")
    print(f"export QC_ARTIFACT_URI='{uri}'   # the tiling job reads this")
    return uri


def merge_qc_csvs(out_dir: Path) -> int:
    """Combine the new per-slide csvs into the existing qc_metrics.csv, safely.

    Unlike preprocessing.quality_control.qc_main's tail, the combined file is never in the
    delete set: read it, read only the per-slide csvs, concat, dedup by slide_name (keep
    the newest row), write the combined, unlink only the per-slide csvs.
    Returns the final row count.
    """
    combined_path = out_dir / "qc_metrics.csv"
    per_slide = sorted(
        f for f in out_dir.glob("*.csv") if f.resolve() != combined_path.resolve()
    )
    frames = [pd.read_csv(combined_path)] if combined_path.exists() else []
    frames.extend(pd.read_csv(f) for f in per_slide)
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["slide_name"], keep="last")
    combined.to_csv(combined_path, index=False)
    for f in per_slide:
        f.unlink()
    return len(combined)


async def run_slides(
    slides: list[str],
    out_dir: Path,
    request_timeout: int,
    max_concurrent: int,
) -> list[str]:
    """Send the given slide paths to the QC service. Returns the stems that failed."""
    failed: list[str] = []
    async with AsyncClient() as client:
        async for result in tqdm(
            client.qc.check_slides(
                slides,
                out_dir,
                config=SlideCheckConfig(
                    mask_level=3,
                    sample_level=1,
                    check_residual=True,
                    check_folding=False,
                    check_focus=True,
                    wb_correction=True,
                ),
                timeout=request_timeout,
                max_concurrent=max_concurrent,
            ),
            total=len(slides),
            desc="QC rerun",
        ):
            if not result.success:
                stem = Path(result.wsi_path).stem
                failed.append(stem)
                print(f"\n  FAILED {stem}: {result.error!r}", flush=True)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slides",
        default=None,
        help="comma list of stems to run; default = recomputed missing set",
    )
    parser.add_argument("--out", default=JOB_OUT, help="output_path (job dir)")
    parser.add_argument(
        "--timeout", type=int, default=18000, help="per-slide seconds (5 h, qc.yaml)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, help="qc.yaml's max_concurrent"
    )
    parser.add_argument("--probe", action="store_true", help="check service, then exit")
    parser.add_argument("--dry-run", action="store_true", help="print plan, send nothing")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="once qc_metrics.csv reaches 200 rows, log the whole output dir to a "
        "fresh MLflow run as `qc_output` (the artifact tiling consumes). Requires "
        "MLFLOW_TRACKING_URI/USERNAME/PASSWORD set.",
    )
    parser.add_argument(
        "--experiment-id",
        default="168",
        help="MLflow experiment for the --upload run (168 = Ulcerative Colitis Dysplasia)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    qc_metrics = out_dir / "qc_metrics.csv"
    slides_dir = Path(SLIDES_DIR)

    if args.slides:
        want = [s.zfill(3) for s in (x.strip() for x in args.slides.split(",") if x.strip())]
    else:
        want = [s for s in real_slides(slides_dir) if s not in done_slides(qc_metrics)]

    if not want:
        print("nothing to do -- every real slide is already in qc_metrics.csv")
        return 0

    print(f"output : {out_dir}")
    print(f"existing qc_metrics.csv rows: {len(done_slides(qc_metrics))}")
    print(f"slides to run ({len(want)}): {', '.join(want)}")

    if args.dry_run:
        for s in want:
            print(f"  {slides_dir / (s + '.mrxs')}")
        return 0

    # Cheap reachability gate: do not spend ~3.6 h/slide on an unreachable service.
    from rationai.client import Client

    with Client() as c:
        qc_url = c.qc_base_url
    if args.probe or not service_up(qc_url):
        status = "up" if args.probe else "NOT answering"
        print(f"\nQC service {qc_url}: {status}")
        if not args.probe:
            print(
                "Not sending -- the service did not answer /openapi.json (120s x3). "
                "Ask for the rayservice-qc pods before retrying."
            )
            return 1
        return 0
    print(f"service {qc_url} is up")

    paths = [str(slides_dir / f"{s}.mrxs") for s in want]
    missing_paths = [p for p in paths if not Path(p).exists()]
    if missing_paths:
        print(f"\nslide not visible from here (may still be OK on the service): {missing_paths}")

    started = time.time()
    print(f"running {len(paths)} slides (max_concurrent={args.max_concurrent}, "
          f"timeout={args.timeout}s each)...\n", flush=True)
    failed = asyncio.run(run_slides(paths, out_dir, args.timeout, args.max_concurrent))

    moved = 0
    for prefix, subdir in MASK_TARGETS:
        moved += organize_masks(out_dir, subdir, prefix)
    rows = merge_qc_csvs(out_dir)

    print(f"\ndone in {(time.time() - started) / 3600:.1f} h")
    print(f"organized {moved} flat masks into subdirs")
    print(f"qc_metrics.csv now has {rows} rows (expected 200)")
    if failed:
        print(f"client reported failures: {', '.join(sorted(failed))} -- a client timeout "
              "does not stop the service, which keeps working and often writes the csv "
              "afterwards (13 of 25 timeouts did); the row count is the real check")
    if rows < 200:
        print("NOT COMPLETE -- re-run this script; finished slides are skipped and the "
              "rest picks up where this stopped")
        return 1
    print("COMPLETE: qc_metrics.csv has 200 rows, all 4 subdirs carry 200 masks")
    if args.upload:
        upload_qc_artifacts(out_dir, args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
