r"""Tile the 186 QC-complete hunCRC slides, from a notebooks-ns job, straight to disk.

    python scripts/run_tiling_186_notebook.py               # all 186 -> /mnt/projects/huncrc_dysplasia/tiling_186
    python scripts/run_tiling_186_notebook.py --smoke       # ONE slide (001) wiring check, ~2-5 min
    python scripts/run_tiling_186_notebook.py --slides 001,002 --out /mnt/projects/huncrc_dysplasia/tiling_x

Why this script instead of `python -m preprocessing.tiling`:
  * NO MLFLOW. The notebooks job has no tracking credentials; the stage's main() is
    wired around artifact URIs (download qc/tissue, upload tiling). Here everything is
    a path on the two shared mounts, and the output lands in a plain directory the
    embeddings job reads directly (scripts/run_embeddings_pilot.py --tiling-dir).
  * THE 14 MISSING QC SLIDES ARE SKIPPED, not fatal. The stage KeyErrors on a slide
    absent from qc_metrics.csv (tiling.py:54 df.loc[stem]); here the slide list is
    built from qc_metrics.csv itself, so the run covers exactly the 186 that have it.
  * TISSUE MASKS CANNOT BE DOWNLOADED (same reason, no auth; the 200-mask run is in
    MLflow, the share holds only 001-003). The QC per-pixel COVERAGE mask
    (quality_control/artifacts_per_pixel/{stem}.tiff, one per QC-complete slide) is a
    real, same-resolution stand-in: it gates tiles the way a tissue mask does and
    fills the `epithelium` column, exactly as the tissue-as-epithelium stand-in in
    the stage does. When the real tissue + episeg masks exist, re-run tiling with
    them (one-folder swap) and the columns go live.

The tiling computation is the UNMODIFIED preprocessing.tiling.tiling() (same code the
job/stage runs), so the output is identical to what the full pipeline would produce for
these 186 -- same 224px @ 0.52 mpp level-2 grid, stride 112, threshold 0.5, same
lowgrade/highgrade annotation polygons from /mnt/projects/huncrc_dysplasia/annotations.

OUTPUT (consumed by run_embeddings_pilot.py --tiling-dir):
    {out}/slides/*.parquet   one parquet, columns incl. id (hash), path, level, mpp_*
    {out}/tiles/*.parquet    batched parquets, columns: slide_id,x,y,tissue,annotation,
                             epithelium,blur,artifacts,clarity,fold,lowgrade_dysplasia,
                             highgrade_dysplasia
    {out}/manifest.json      slide/tile counts + params, for the monitor

PROGRESS: a tqdm bar updated per 100k-tile batch (watch `kubectl -n rationai-notebooks-ns
logs -f <pod>`) plus a per-batch line with the running tiles/s rate. Pilot reference:
3 slides = 36,170 tiles in ~67 min wall (~540 tiles/min), so the 186-slide run
(~2.2M tiles) ETA is on the order of ~70 h at that rate -- the per-batch rate is the
number to watch; the work is CPU-bound (shapely + vips overlap, no GPU), so a higher
cpu request on the job pod scales it down linearly.

Needs both shared mounts inside the pod (the notebooks-ns Jupyter has them, so a job
in the same ns with the same volumes does):
    /mnt/data/Public/...   (the .mrxs slides, opened by read_slides)
    /mnt/projects/huncrc_dysplasia/   (qc dir, annotations, output)
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

#: Same scripts/ shadow rule as run_tiling_pilot.py: repo root first so
#: `preprocessing` binds the stage package, not scripts/preprocessing/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq
import ray

from preprocessing.tiling import tiling

SLIDES_DIR = (
    Path("/mnt/data/Public/Public/colorectum/TCIA_HunCRC")
    / "PKG - Hungarian-Colorectal-Screening_v2"
)
QC_DIR = Path("/mnt/projects/huncrc_dysplasia/quality_control")
ANNOT_DIR = Path("/mnt/projects/huncrc_dysplasia/annotations")
DEFAULT_OUT = Path("/mnt/projects/huncrc_dysplasia/tiling_186")

#: Proven pilot geometry (run_tiling_pilot.py): level 2 of the 0.121267 mpp pyramid.
MPP = 0.52
TILE_EXTENT = 224
STRIDE = 112
TISSUE_THRESHOLD = 0.5
TARGET_GROUPS = ["lowgrade_dysplasia", "highgrade_dysplasia"]
BATCH_ROWS = 100_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides", default="", help="comma list of stems; default = all in qc_metrics")
    parser.add_argument("--smoke", action="store_true",
                        help="ONE slide (001) to a -smoke dir: wiring check before the long run")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(str(DEFAULT_OUT) + "-smoke") if args.smoke else DEFAULT_OUT
    )

    #: The slide list IS the QC coverage: stems present in qc_metrics.csv.
    qc_metrics = pd.read_csv(QC_DIR / "qc_metrics.csv")
    done = {str(int(s)).zfill(3) for s in qc_metrics["slide_name"] if str(s).lstrip("-").isdigit()}
    if args.slides:
        want = [s.zfill(3) for s in (x.strip() for x in args.slides.split(",") if x.strip())]
        missing_qc = [s for s in want if s not in done]
        if missing_qc:
            raise SystemExit(f"these slides have no qc_metrics row, the stage would KeyError: {missing_qc}")
    elif args.smoke:
        want = ["001"]
    else:
        want = sorted(done)

    paths = []
    skipped = []
    for stem in want:
        slide = SLIDES_DIR / f"{stem}.mrxs"
        annot = ANNOT_DIR / f"{stem}.json"
        if not slide.exists():
            skipped.append((stem, "no .mrxs"))
        elif not annot.exists():
            skipped.append((stem, "no annotation json"))
        else:
            paths.append(slide)
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    if not paths:
        raise SystemExit("no slides to tile")

    #: Coverage mask as the tissue stand-in (see module docstring). Blur mask from its
    #: own subdir; both exist for every QC-complete slide.
    tissue_folder = QC_DIR / "artifacts_per_pixel"
    n_tissue = len(list(tissue_folder.glob("*.tiff")))
    print(f"tiling {len(paths)} slides -> {out_dir}")
    print(f"qc rows: {len(done)}, coverage-mask stand-in present: {n_tissue}")
    print(f"geometry: {TILE_EXTENT}px @ {MPP} mpp, stride {STRIDE}, threshold {TISSUE_THRESHOLD}")
    print(f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.DataFrame({"slide_path": [str(p) for p in paths]})

    ray.init(runtime_env={"excludes": [".git", ".venv"]})
    try:
        t0 = time.time()
        slides_ds, tiles_ds = tiling(
            df,
            qc_folder=QC_DIR,
            tissue_folder=tissue_folder,
            epithelium_folder=tissue_folder,
            annot_folder=ANNOT_DIR,
            tile_extent=TILE_EXTENT,
            stride=STRIDE,
            mpp=MPP,
            tissue_threshold=TISSUE_THRESHOLD,
            target_groups=TARGET_GROUPS,
        )

        print("materializing slides (openslide metadata + qc join)...", flush=True)
        slides_df = slides_ds.to_pandas()
        (out_dir / "slides").mkdir(parents=True, exist_ok=True)
        slides_df.to_parquet(out_dir / "slides" / "slides.parquet")
        id_to_stem = dict(zip(slides_df["id"], [Path(p).stem for p in slides_df["path"]]))
        print(
            f"slides done in {(time.time() - t0) / 60:.1f} min "
            f"({len(slides_df)} rows) -> {out_dir / 'slides' / 'slides.parquet'}",
            flush=True,
        )

        #: PROGRESS: one line per 100k-tile batch (a tqdm bar renders badly in
        #: `kubectl logs` and would jump ~3 h at a time anyway). The ETA is a
        #: lower bound: tiles stream in arrival order (slides interleaved within a
        #: 4096-row block), so "slides seen" lags the true fraction tiled.
        slides_expected = len(paths)
        t1 = time.time()
        tiles_dir = out_dir / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        n_tiles = 0
        per_slide = Counter()
        i = 0
        for batch in tiles_ds.iter_batches(batch_size=BATCH_ROWS, batch_format="pyarrow"):
            n = batch.num_rows
            per_slide.update(batch["slide_id"].to_pylist())
            #: iter_batches(batch_format="pyarrow") yields pyarrow Tables -- write as-is.
            pq.write_table(batch, tiles_dir / f"tiles_{i:04d}.parquet")
            n_tiles += n
            i += 1
            elapsed = max(time.time() - t1, 1)
            rate = n_tiles / elapsed
            slides_seen = len(per_slide)
            eta_h = (slides_expected - slides_seen) / max(slides_seen / elapsed, 1e-9) / 3600
            print(
                f"[{time.strftime('%H:%M:%S')}] {slides_seen}/{slides_expected} slides seen, "
                f"{n_tiles:,} tiles, {rate:.0f} tiles/s, ETA ~{eta_h:.1f} h  "
                f"(batch {i} -> tiles_{i - 1:04d}.parquet)",
                flush=True,
            )

        manifest = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.time() - t1),
            "slides": len(slides_df),
            "tiles": n_tiles,
            "tiles_per_slide": {
                id_to_stem[k]: v for k, v in sorted(per_slide.items(), key=lambda kv: id_to_stem[kv[0]])
            },
            "params": {
                "mpp": MPP, "tile_extent": TILE_EXTENT, "stride": STRIDE,
                "tissue_threshold": TISSUE_THRESHOLD,
                "target_groups": TARGET_GROUPS,
                "tissue_standin": str(tissue_folder),
            },
            "qc_rows_at_start": len(done),
            "slides_skipped": skipped,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nDONE in {(time.time() - t1) / 3600:.1f} h: {n_tiles:,} tiles / {len(slides_df)} slides")
        print(f"output: {out_dir}  (manifest: {out_dir / 'manifest.json'})")
        print("next: uv run python scripts/run_embeddings_pilot.py --tiling-dir "
              f"{out_dir} --out /mnt/projects/huncrc_dysplasia/embeddings_186")
        return 0
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
