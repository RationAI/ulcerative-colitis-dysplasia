"""Tiling pilot for the 3 HunCRC slides we already have all inputs for.

    uv run python scripts/run_tiling_pilot.py --dry-run     # preflight only
    uv run python scripts/run_tiling_pilot.py                # 3 slides, real run

WHY THIS SCRIPT AND NOT `python -m preprocessing.tiling`. Three blockers would stop the
pipeline stage cold for this cohort, and a 3-slide pilot has no business depending on
config surgery:

  1. QC METRICS INDEX TYPE. preprocessing/tiling.py:216 reads
     `qc_metrics.csv` with `index_col="slide_name"`, and the index comes back int64
     (it was written as bare integers because `slide_name` values are digits). But
     `qc_agg` at :31 does `df.loc[Path(row["path"]).stem]` where stem is a STRING
     "001", so `loc["001"]` raises KeyError. This is a real bug for the HunCRC
     cohort -- it doesn't affect IBD where paths are like "NN_NN_HE_X" (strings).
     Workaround here: convert the index to zero-padded string index before calling
     tiling().

  2. MISSING SPLITS AND processed_w_epi.yaml. Tiling.main() expects
     `config.dataset.mlflow_uris.splits` (a dict name -> artifact URI) and a
     `config.dataset.epithelium` dir. The dataset config
     `configs/dataset/processed_w_masks.yaml` references `_dataset/processed_w_epi@_here_`
     which does not exist on this branch, and its qc/splits URIs are experiment 111
     (IBD). We bypass all of it: for 3 slides we load just those slides as a "split".

  3. EPITHELIUM INPUT PATH. Tiling needs per-slide epithelium masks at
     `{epithelium_folder}/{stem}.tiff`. The IBD path is a dataset config key; the HunCRC
     pilot writes to the local run dir (`raw_pilot`) so the new script can feed that
     directly.

WHAT IT RUNS. The function `preprocessing.tiling.tiling(df, ...)` unchanged: for each
slide, it grids over the slide extent at --mpp / --tile-extent / --stride, rasterises
the annotation polygons via the ratiopath EMPAIA parser, sums up the
tissue/blur/artifacts/epithelium mask coverage over each tile, and returns two Ray
Datasets (`slides`, `tiles`). The output is two parquet directories:

    {out}/slides/<...>.parquet         one row per slide (geometry + QC + mask paths)
    {out}/tiles/<...>.parquet          one row per tile after tissue >= threshold filter

PARAMS. 224px tiles at mpp 0.52 (resolves to HunCRC level 2, 0.4851 -- the closest
pyramid level to the IBD 0.52 scale) with stride 112 (50% overlap). This matches the
UC-dysplasia model's IBD input scale (tiled/level1_extent224.yaml: 224px @ 0.52, stride
112) so its pretrained weights transfer, rather than the HunCRC label grid (level 1,
0.2425). 224px @ 0.4851 = ~109um FOV, vs IBD's 224px @ 0.52 = ~117um. The IBD 1.55
second scale has no clean HunCRC level (nearest is level 4, 1.94), so it is deferred.
Change them to re-tile at another resolution -- note a new (tile_extent, stride, mpp)
creates a NEW tiling, not a filtered view of an old one.

INPUTS DEFAULTED TO DISK, NOT MLFLOW, because the HunCRC blobs aren't uploaded to MLflow
yet:
  --slides-dir          /mnt/data/.../PKG - Hungarian-Colorectal-Screening_v2 (the .mrxs)
  --qc-dir              /mnt/projects/huncrc_dysplasia/quality_control
  --tissue-dir          /mnt/projects/huncrc_dysplasia/tissue_masks
  --epithelium-dir      /mnt/projects/huncrc_dysplasia/epithelium_masks/raw_pilot
  --annot-dir           /mnt/projects/huncrc_dysplasia/annotations
  --out                 /mnt/projects/huncrc_dysplasia/tiling_pilot

Use --dry-run first. It prints what inputs exist, validation results, and a summary of
planned tile counts. Nothing is written and no Ray is started until the real run.
"""

import argparse
import sys
import time
import pandas as pd
from pathlib import Path


SLIDES_DIR = Path(
    "/mnt/data/Public/Public/colorectum/TCIA_HunCRC/"
    "PKG - Hungarian-Colorectal-Screening_v2"
)
QC_DIR = Path("/mnt/projects/huncrc_dysplasia/quality_control")
TISSUE_DIR = Path("/mnt/projects/huncrc_dysplasia/tissue_masks")
#: No episeg run yet (epithelium_masks/raw_pilot is empty), so the tissue masks double
#: as the epithelium input -- same {stem}.tiff shape. Repoint at the real episeg dir
#: later; nothing else changes.
EPITHELIUM_DIR = Path("/mnt/projects/huncrc_dysplasia/tissue_masks")
ANNOT_DIR = Path("/mnt/projects/huncrc_dysplasia/annotations")
OUT_DIR = Path("/mnt/projects/huncrc_dysplasia/tiling_pilot")

DEFAULT_SLIDES = ["001", "002", "003"]
#: IBD's first scale (0.52 mpp). read_slides snaps to the nearest HunCRC pyramid level,
#: which is level 2 (0.4851) -- the closest available, ~7% off, so this is a clean
#: match for the UC-dysplasia model's 0.52-scale input rather than level 1 (0.2425),
#: the label grid, which would be 2x too fine for these weights.
DEFAULT_MPP = 0.52
#: Model input size (224px, the IBD convention). Tiling at a different (extent, stride,
#: mpp) creates a NEW tiling, NOT a filtered view of an old one.
TILE_EXTENT = 224
#: IBD stride 112 = 50% overlap. Use 224 to halve the tile count instead.
STRIDE = 112
#: tissue_threshold from configs/preprocessing/tiling.yaml -- the fraction of the
#: central half-ROI required to be tissue.
TISSUE_THRESHOLD = 0.5

#: Annotation label names as they literally appear inside the HunCRC EMPAIA JSONs
#: (verified for 001/002/003: 'lowgrade_dysplasia', 'highgrade_dysplasia', 'normal',
#: 'inflammation'). tiling() compiles ^name$ regexes from these, so they must match
#: the JSON names exactly -- the IBD config values "LG Dysplasia"/"HG Dysplasia"
#: would silently produce zero annotation coverage. This drives the tiles-parquet
#: column names, which the tile-masks pilot then reads back via --percentage-cols.
TARGET_GROUPS = ["lowgrade_dysplasia", "highgrade_dysplasia"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slides", default=",".join(DEFAULT_SLIDES),
                    help="comma-separated slide stems")
    ap.add_argument("--slides-dir", type=Path, default=SLIDES_DIR)
    ap.add_argument("--qc-dir", type=Path, default=QC_DIR)
    ap.add_argument("--tissue-dir", type=Path, default=TISSUE_DIR)
    ap.add_argument("--epithelium-dir", type=Path, default=EPITHELIUM_DIR)
    ap.add_argument("--annot-dir", type=Path, default=ANNOT_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--tile-extent", type=int, default=TILE_EXTENT)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--mpp", type=float, default=DEFAULT_MPP)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs and describe outputs; don't run")
    args = ap.parse_args()

    slide_stems = [s.strip() for s in args.slides.split(",")]

    #: ---- preflight ----
    print("1. preflight\n")
    issues = 0

    for d in [args.slides_dir, args.qc_dir, args.tissue_dir, args.epithelium_dir, args.annot_dir]:
        if not d.is_dir():
            print(f"  MISSING DIR: {d}")
            issues += 1
    if issues:
        print(f"\n{issues} missing input dirs. Fix before running.")
        return 1

    dataset_rows = []
    for stem in slide_stems:
        slide_path = args.slides_dir / f"{stem}.mrxs"
        if not slide_path.is_file():
            print(f"  MISSING SLIDE: {slide_path}")
            issues += 1
            continue

        tissue = args.tissue_dir / f"{stem}.tiff"
        epi = args.epithelium_dir / f"{stem}.tiff"
        annot = args.annot_dir / f"{stem}.json"
        qc_blur = args.qc_dir / "blur_per_pixel" / f"{stem}.tiff"
        qc_artifacts = args.qc_dir / "artifacts_per_pixel" / f"{stem}.tiff"

        for label, p in [("tissue", tissue), ("epi", epi), ("annot", annot),
                         ("qc_blur", qc_blur), ("qc_artifacts", qc_artifacts)]:
            if not p.is_file():
                print(f"  {stem}: MISSING {label}: {p}")
                issues += 1

        dataset_rows.append({"slide_path": str(slide_path), "stem": stem})

    # QC metrics index fix -- the real bug for HunCRC
    qc_csv = args.qc_dir / "qc_metrics.csv"
    if not qc_csv.is_file():
        print(f"  MISSING QC METRICS: {qc_csv}")
        issues += 1
    else:
        #: Same int64-index / string-stem fix the staging step below applies to the
        #: file tiling() actually reads (the slide_name column is numeric); without it
        #: the preflight's string lookup KeyErrors and reports slides as unprocessed.
        qc_df = pd.read_csv(qc_csv, index_col="slide_name")
        qc_df.index = qc_df.index.astype(str).str.zfill(3)
        for stem in slide_stems:
            try:
                _ = qc_df.loc[stem, "mean_coverage(Piqe)"]
            except KeyError:
                print(
                    f"  {stem}: NOT IN QC METRICS AFTER ZERO-PADDING. "
                    "This means the QC stage never processed this slide. "
                    "Recovery here: skip, or add to --slides."
                )
                issues += 1

    if issues:
        print(f"\n{issues} preflight failures. Nothing was run.")
        return 1

    print("  all inputs verified for slides:", ", ".join(slide_stems))
    print(f"\n2. tiling parameters: mpp={args.mpp}, tile_extent={args.tile_extent}, "
          f"stride={args.stride}, tissue_threshold={TISSUE_THRESHOLD}")
    print(f"  out: {args.out}")
    print(f"  target_groups: {TARGET_GROUPS}  "
          "(snake_case names as used inside the EMPAIA annotation JSONs; the IBD "
          "config's 'LG Dysplasia'/'HG Dysplasia' would match nothing here)")
    print(f"\n  EXPECTED: one row per tile > tissue_threshold in the {OUT_DIR}/tiles parquet.")
    print(f"  001.mrxs at level 0: 545536x588800; tiles ~= "
          f"{(545536 // args.stride) * (588800 // args.stride)} raw spots; "
          f"most discarded by tissue threshold.")

    if args.dry_run:
        print("\ndry run done.")
        return 0

    print(f"\n3. starting tiling (Ray)\n", flush=True)
    started = time.time()

    #: The surgical part: build the "split" df the same shape as the parquet-run would
    #: produce. Columns must include slide_path AND the zero-padded stem alias
    #: used by Path(row["path"]).stem in the pipeline's dictops.
    dataset = pd.DataFrame(dataset_rows)

    #: WORKAROUND for the QC metrics int/string index mismatch. Tiling.tiling() reads
    #: qc_folder / "qc_metrics.csv" with `index_col="slide_name"` and does `df.loc[stem]`
    #: with a string. Our CSV's slide_name column is numeric, so this would fail.
    #: Build a staging directory that mimics what the QC stage writes, with a re-indexed
    #: qc_metrics.csv and symlinked mask subdirs. Nothing is copied; symlinks mean the
    #: real tiffs stay in place, and any future fix to the QC csv just regenerates this.
    staging_qc = args.out / "_qc_staging"
    staging_qc.mkdir(parents=True, exist_ok=True)
    qc_df_fixed = pd.read_csv(qc_csv, index_col="slide_name")
    qc_df_fixed.index = qc_df_fixed.index.astype(str).str.zfill(3)
    qc_df_fixed.to_csv(staging_qc / "qc_metrics.csv")
    for sub in ["blur_per_pixel", "artifacts_per_pixel"]:
        src = args.qc_dir / sub
        dst = staging_qc / sub
        if not dst.exists():
            try:
                dst.symlink_to(src, target_is_directory=True)
            except OSError:
                #: symlink failed (maybe permission) -- fall back to a plain copy
                import shutil

                shutil.copytree(src, dst)

    #: From `python scripts/x.py`, sys.path[0] is scripts/, and scripts/preprocessing
    #: (the kube submitters) is a namespace package that shadows the top-level regular
    #: `preprocessing` package -- so `from preprocessing.tiling import tiling` would land
    #: on scripts/preprocessing/tiling.py, a submitter that calls submit_job at import.
    #: Prepend the repo root so the real package wins. Same footgun the tissue script
    #: sidesteps by loading by path (scripts/run_tissue_masks_locally.py:88).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preprocessing.tiling import tiling

    models_dir = args.epithelium_dir
    ds_slides, ds_tiles = tiling(
        dataset,
        qc_folder=staging_qc,
        tissue_folder=args.tissue_dir,
        epithelium_folder=models_dir,
        annot_folder=args.annot_dir,
        tile_extent=args.tile_extent,
        stride=args.stride,
        mpp=args.mpp,
        tissue_threshold=TISSUE_THRESHOLD,
        target_groups=TARGET_GROUPS,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    slides_out = args.out / "slides"
    tiles_out = args.out / "tiles"
    print(f"\n4. writing slides parquet to {slides_out}")
    ds_slides.write_parquet(str(slides_out), min_rows_per_file=50_000)
    print(f"5. writing tiles parquet to {tiles_out}")
    ds_tiles.write_parquet(str(tiles_out), min_rows_per_file=50_000)

    elapsed = time.time() - started
    print(f"\ndone in {elapsed:.0f}s")
    print(f"slides rows: ", end="")
    n_slides = ds_slides.count()
    print(n_slides)
    n_tiles = ds_tiles.count()
    print(f"tiles rows: {n_tiles}")

    if n_tiles == 0:
        print("\nNO TILES SURVIVED the tissue filter. Check --tissue-dir and the mask\n"
              "geometry; the tissue mask needs to be level-3 (matches the 68192x73600 shape\n"
              "openslide reports) or the geometry mapping below will be wrong.")
        return 1

    print(f"\nslides parquet: {slides_out}")
    print(f"tiles parquet: {tiles_out}")
    print("\nNext: the tile-mask stage (run_tile_masks_pilot.py) reads these parquets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
