"""Tile-masks pilot: render the 3-slide tiling results into per-label TIFFs.

    uv run python scripts/run_tile_masks_pilot.py --dry-run     # preflight only
    uv run python scripts/run_tile_masks_pilot.py                # real run

WHY THIS SCRIPT AND NOT `python -m preprocessing.tile_masks`. Two blockers in the
pipeline stage, same family as in run_tiling_pilot.py:

  1. MISSING FILTERED-TILING URI. Tile-masks.main() reads
     `config.dataset.mlflow_uris.tiling_filtered`, a mapping split_name -> artifact URI
     (`configs/dataset/tiled_filtered/...`) and downloads all three splits to build up the
     slides/tiles for rendering. For HunCRC we have ONE pilot tiling written on disk by
     scripts/run_tiling_pilot.py, not in MLflow, so we load it directly.

  2. KEY NAME MISMATCH. In the repo stage, process_slide expects a pd.Series keyed by
     `slide["id"]` and does `tiles[tiles["slide_id"] == slide["id"]]` (:28). The slides
     parquet written by `preprocessing.tiling.tiling()` gets its `id` from
     `ratiopath.tiling.utils.row_hash` (a SHA256 hex), and `tiles.slide_id` is the SAME
     hash. This script reads slides parquet and hands `slide["id"]` straight through,
     so the lookup matches -- no translation needed.

WHAT IT RUNS. The ACTUAL `preprocessing.tile_masks.process_slide` (:20-113), unchanged,
via `ray.get(process_slide.remote(...))`. Each slide becomes one task that
  (a) converts tile coords from level-0 pixels into the output mask's level-3 pixels
      via the MPP ratio, snapped to the output stride grid,
  (b) keeps only tiles fitting in the level-3 mask extent,
  (c) builds one scalar mask per --percentage-col via ratiopath.mask_builders.MaskBuilder,
      scaled 0..255, and writes per-col TIFF pyramids into {out}/{col}/{stem}.tiff,
  (d) draws the 'outlines' mask marking tile boundaries,
  (e) writes {out}/outlines/{stem}.tiff.
process_slide is NOT decorated with @ray.remote(memory=...), but Ray's default
Memory=0 (arguments only) means Ray will schedule it as a normal task. Because the
previous pilot script runs were 3-4 GiB per slide, we call it with --workers 1
(default) -- you can raise it with a flag.

INPUTS:
    --tiling-dir          directory containing slides/ and tiles/ parquet dirs,
                          i.e. the output_dir of scripts/run_tiling_pilot.py

OUTPUTS:
    {out}/outlines/{stem}.tiff
    {out}/{percentage_col}/{stem}.tiff          one per percentage_col
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

TILING_PILOT = Path("/mnt/projects/huncrc_dysplasia/tiling_pilot")
OUT_DIR = Path("/mnt/projects/huncrc_dysplasia/tile_masks_pilot")
#: Percentage columns to rasterise. Must be a subset of the tiles-parquet columns and
#: therefore reflect the labels the tiling pilot ran with: HunCRC annotation JSONs use
#: snake_case names, so the pilot emits lowgrade_dysplasia/highgrade_dysplasia columns
#: (the IBD config's "LG Dysplasia"/"HG Dysplasia" would match nothing). Override with
#: --percentage-cols if you re-tile with different target_groups.
PERCENTAGE_COLS = ["epithelium", "blur", "artifacts",
                   "lowgrade_dysplasia", "highgrade_dysplasia"]
#: Output pyramid level -- **2, not the yaml's 3.** Level 3 upsamples the tiling
#: (which is at level 2, 0.4851 mpp): the mask becomes ~68k x 73k px and its float32
#: MaskBuilder accumulator is ~20 GB, OOM-ing under the 32 GB pod cap.
#: Level 2 matches the tiling 1:1 -> mask ~17k x 18k px, accumulator ~1.2 GB.
LEVEL = 2
#: Memory reservation per task. process_slide is NOT decorated with a memory option in
#: the repo (preprocessing/tile_masks.py:20-27), but it opens the slide (~545k x 588k
#: pixels via OpenSlide) and builds multiple mask accumulator arrays in memory.
#: Measured on the full-cohort job (2026-10-22, tiling_186_merged): a single
#: ray::process_slide peaked at 10.9 GiB, so the old 4 GiB reservation let two of them
#: run inside a 16Gi pod and Ray's OOM monitor killed both at 15.23GB / 16.00GB.
MEMORY_PER_TASK = 12 * 1024**3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiling-dir", type=Path, default=TILING_PILOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--level", type=int, default=LEVEL)
    ap.add_argument("--percentage-cols", nargs="+", default=PERCENTAGE_COLS,
                    help="columns of the tiles parquet to rasterise, one TIFF set each")
    ap.add_argument("--workers", type=int, default=1,
                    help="max_concurrent Ray tasks; each opens a slide")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    percentage_cols = args.percentage_cols

    if not args.tiling_dir.is_dir():
        sys.stderr.write(
            f"tiling pilot {args.tiling_dir} not found. Run scripts/run_tiling_pilot.py first.\n"
        )
        return 1
    slides_path = args.tiling_dir / "slides"
    tiles_path = args.tiling_dir / "tiles"
    if not slides_path.is_dir() or not tiles_path.is_dir():
        sys.stderr.write(
            f"expected slides/ and tiles/ parquet subdirs under {args.tiling_dir}; "
            f"got slides={slides_path.is_dir()} tiles={tiles_path.is_dir()}\n"
        )
        return 1

    print("1. preflight")
    try:
        slides = pd.read_parquet(slides_path)
        tiles = pd.read_parquet(tiles_path)
    except Exception as e:
        sys.stderr.write(f"failed reading parquet: {e}\n")
        return 1

    #: columns slides.parquet MUST have for process_slide:
    req_cols = {"path", "id", "tile_extent_x", "tile_extent_y", "mpp_x", "mpp_y"}
    missing = req_cols - set(slides.columns)
    if missing:
        sys.stderr.write(
            f"slides parquet missing columns {missing}; got {list(slides.columns)}.\n"
            "The tilt parquet written by scripts/run_tiling_pilot.py is expected to have\n"
            "at least: path, extent_x, extent_y, tile_extent_x, tile_extent_y, stride_x,\n"
            "stride_y, mpp_x, mpp_y, level, downsample, id, plus blur_mean/artifacts_mean.\n"
        )
        return 1
    req_tcols = {"slide_id", "x", "y"} | set(percentage_cols)
    missing_t = req_tcols - set(tiles.columns)
    if missing_t:
        sys.stderr.write(
            f"tiles parquet missing columns {missing_t}; got {list(tiles.columns)}.\n"
        )
        return 1

    #: id <-> slide consistency check
    id_to_path = dict(zip(slides["id"], slides["path"]))
    tiles_per_id = tiles.groupby("slide_id").size().rename("n_tiles").reset_index()
    orphans = [sid for sid in tiles_per_id["slide_id"] if sid not in id_to_path]
    if orphans:
        sys.stderr.write(
            f"tiles reference slide_ids with no matching slides row: {orphans[:5]}\n"
        )
        return 1

    print(f"   slides rows: {len(slides)}\n")

    if args.dry_run:
        print("dry run done.")
        for _, row in slides.iterrows():
            stem = Path(row["path"]).stem
            n = int((tiles["slide_id"] == row["id"]).sum())
            print(f"   {stem}: {n} tiles")
        return 0

    #: Pre-make the output dirs, same shape as the repo stage. Do NOT wipe anything
    #: that already exists: pilots are resumable only because an existing tiff at
    #: {out}/{col}/{stem}.tiff is accepted as done for that slide+col.
    args.out.mkdir(parents=True, exist_ok=True)
    for col in [*percentage_cols, "outlines"]:
        (args.out / col).mkdir(parents=True, exist_ok=True)

    print(f"\n2. ray.init(workers={args.workers})")
    import ray

    with ray.init(
        num_cpus=max(args.workers, 1),
        #: Ray would otherwise take ~30% of the pod cgroup for the object store,
        #: and the OOM monitor counts it against the same limit as the workers.
        object_store_memory=2 * 1024**3,
        include_dashboard=False,
        configure_logging=False,
        runtime_env={"excludes": [".git", ".venv"]},
    ):  # type: ignore[arg-type]
        #: Same scripts/preprocessing shadow as run_tiling_pilot.py: prepend the repo
        #: root so we bind the actual `preprocessing/tile_masks`, not the submitter.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from preprocessing.tile_masks import process_slide

        refs: list = []
        names: list[str] = []
        per_slide_tiles: list[pd.DataFrame] = []

        for _, slide_row in slides.iterrows():
            slide_id = slide_row["id"]
            slide_tiles = tiles[tiles["slide_id"] == slide_id]
            print(
                f"   {Path(slide_row['path']).stem}: {len(slide_tiles)} tiles"
            )
            #: process_slide decorated via @ray.remote with no memory option; apply
            #: explicit reservation so repeated tasks don't OOM.
            ref = process_slide.options(memory=MEMORY_PER_TASK).remote(
                slide_row,
                percentage_cols,
                args.out,
                slide_tiles,
                args.level,
            )
            refs.append(ref)
            names.append(Path(slide_row["path"]).stem)
            per_slide_tiles.append(slide_tiles)

        print(f"\n3. awaiting {len(refs)} task(s)...")
        started = time.time()
        results = []
        for stem, ref in zip(names, refs):
            try:
                _ = ray.get(ref, timeout=3600)
                results.append((stem, True, time.time()))
                print(f"   {stem}: ok", flush=True)
            except Exception as e:
                results.append((stem, False, str(e)))
                print(f"   {stem}: FAILED {e}", flush=True)

    total = time.time() - started
    print(f"\nfinished Ray in {total:.0f}s")

    print(f"\n4. verifying outputs under {args.out}")
    for stem, ok, _ in results:
        outlines_path = args.out / "outlines" / f"{stem}.tiff"
        pct_paths = [args.out / col / f"{stem}.tiff" for col in percentage_cols]
        missing_cols = [col for col, p in zip(percentage_cols, pct_paths) if not p.is_file()]
        if not ok:
            print(f"   {stem}: task failed, skipping")
            continue
        if not outlines_path.is_file() or missing_cols:
            print(
                f"   {stem}: PARTIAL (outlines={outlines_path.is_file()}, "
                f"missing={missing_cols})"
            )
            continue
        print(f"   {stem}: all masks written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
