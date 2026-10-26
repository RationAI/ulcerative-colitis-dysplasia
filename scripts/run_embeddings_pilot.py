"""Tile-embedding pilot for the 3-slide HunCRC tiling, against the model service.

    uv run python scripts/run_embeddings_pilot.py --probe        # one-tile smoke test
    uv run python scripts/run_embeddings_pilot.py --crop-only    # verify crop path (svc down)
    uv run python scripts/run_embeddings_pilot.py --limit 50     # 50 tiles/slide
    uv run python scripts/run_embeddings_pilot.py                # all 36,170 tiles

This is the 3-slide local analogue of the stage `preprocessing/embeddings.py` and
reuses that module's code (`embed_image`, `EmbedTiles`, `crop_tiles`,
`add_embedding_column`, `write_per_slide`) -- the same pilot->stage relationship as
run_tiling_pilot -> preprocessing.tiling. The ONE delta is the input: the stage reads
HunCRC tiling from MLflow URIs; here there is no HunCRC tiling in MLflow, so it reads
the local output of scripts/run_tiling_pilot.py and joins slides onto tiles the same way
the stage does (path, level, tile_extent).

Readiness: prov-gigapath is scaled-to-zero today (same as episeg), so the real run waits
on the huncrc reroll. `--probe` is the readiness check -- it crops one real tile and runs
it end-to-end, replacing the "is the model up?" route-table guess with "did a vector come
back". `--crop-only` proves the tiling->crop half with the service down.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

#: Same scripts/preprocessing shadow as run_tiling_pilot.py: prepend the repo root so we
#: bind the real `preprocessing/embeddings.py`, not scripts/preprocessing/embeddings.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.embeddings import (  # noqa: E402
    add_embedding_column,
    crop_tiles,
    embed_image,
)

#: Target the HUNCRC rerolled deployment, not IBD's rayservice-model-fix-serve-svc.
DEFAULT_SVC = "http://rayservice-model-huncrc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000"
#: IBD TRAINED on prov-gigapath (feat/modeling: embedd_model: prov-gigapath), even though
#: the preprocessing config default says virchow2. Keep the trained model.
DEFAULT_MODEL = "prov-gigapath"
TILING_PILOT = Path("/mnt/projects/huncrc_dysplasia/tiling_pilot")
OUT_DIR = Path("/mnt/projects/huncrc_dysplasia/embeddings_pilot")


def read_tiling(tiling_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    slides = pd.read_parquet(tiling_dir / "slides")
    tiles = pd.read_parquet(tiling_dir / "tiles")
    req = {"id", "path", "level", "tile_extent_x", "tile_extent_y"}
    if missing := (req - set(slides.columns)):
        sys.stderr.write(f"slides parquet missing {missing}; got {list(slides.columns)}\n")
        raise SystemExit(1)
    if not {"slide_id", "x", "y"}.issubset(tiles.columns):
        sys.stderr.write(f"tiles parquet missing geometry; got {list(tiles.columns)}\n")
        raise SystemExit(1)
    return slides, tiles


def enrich(slides: pd.DataFrame, tiles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Join slides onto tiles (path, level, tile_extent), exactly as the stage does."""
    slide_info = slides.set_index("id")[["path", "level", "tile_extent_x", "tile_extent_y"]]
    tiles_enriched = tiles.join(slide_info, on="slide_id")
    id_to_stem = dict(zip(slides["id"], [Path(p).stem for p in slides["path"]]))
    return tiles_enriched, id_to_stem


def expected_counts(tiles: pd.DataFrame, id_to_stem: dict[str, str]) -> dict[str, int]:
    """Tiles per slide stem -- the completion criterion for a flushed parquet."""
    per_slide = tiles.groupby("slide_id").size()
    return {id_to_stem[sid]: int(n) for sid, n in per_slide.items()}


def pod_cpu_limit() -> int:
    """CPUs this pod may actually use, from the cgroup quota.

    `ray.init(num_cpus=...)` must not be handed the connection count: a kuba pod with
    limits.cpu 8 was told it had 32, so Ray scheduled 32 crop tasks onto 8 throttled
    cores, the async embedding actors never got scheduled, and every in-flight request
    sat on a connection until httpx raised PoolTimeout with 0 tiles embedded.
    """
    for quota_path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),                      # cgroup v2: "N 100000"
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            text = Path(quota_path).read_text().split()
            if period_path is None:
                quota, period = text[0], text[1]
                if quota == "max":
                    continue
                return max(1, round(int(quota) / int(period)))
            quota = int(text[0])
            if quota > 0:
                return max(1, round(quota / int(Path(period_path).read_text())))
        except (OSError, ValueError, IndexError):
            continue
    return os.cpu_count() or 1


def already_done(model_dir: Path, counts: dict[str, int]) -> set[str]:
    """Slides whose parquet exists AND holds the full tile count -> resume skips them.

    A parquet with a short row count is a tail flush from a previous run (or a torn
    write); it is redone rather than trusted, so resume can never produce a hole.
    """
    done: set[str] = set()
    for stem, want in counts.items():
        path = model_dir / f"{stem}.parquet"
        if not path.is_file():
            continue
        try:
            got = pq.ParquetFile(path).metadata.num_rows
        except Exception:  # noqa: BLE001 -- a corrupt file is simply redone
            got = -1
        if got == want:
            done.add(stem)
        else:
            print(f"  resume: {stem}.parquet has {got} rows, expected {want} -- redoing")
    return done


def flush_slide(stem: str, df: pd.DataFrame, model_dir: Path) -> Path:
    """Write one slide's parquet atomically, so a kill never leaves a half file."""
    out = model_dir / f"{stem}.parquet"
    tmp = model_dir / f".{stem}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return out


def summary(model_dir: Path, counts: dict[str, int]) -> int:
    """Confirm what is on disk: rows per slide (parquet metadata) and the vector dim."""
    paths = {stem: model_dir / f"{stem}.parquet" for stem in sorted(counts)}
    dim = 0
    for path in paths.values():
        if path.is_file():
            emb = pd.read_parquet(path, columns=["embedding"])
            dim = len(emb["embedding"].iloc[0])
            break
    missing = []
    for stem, path in paths.items():
        if not path.is_file():
            missing.append(stem)
            print(f"  {stem}: MISSING")
            continue
        rows = pq.ParquetFile(path).metadata.num_rows
        flag = "" if rows == counts[stem] else f"  (expected {counts[stem]})"
        print(f"  {stem}: {rows} tiles, dim {dim} -> {path}{flag}")
    if missing:
        sys.stderr.write(f"{len(missing)} slide(s) still missing: {', '.join(missing)}\n")
        return 1
    return 0


def probe(svc: str, model: str, tiling_dir: Path) -> int:
    """Crop ONE real tile and run it through the service end-to-end.

    The route table lists deployments with no replica (episeg, prov-gigapath today), so
    "is the model up" is answered only by getting a vector back. A dead replica 5xxs or
    times out here, it does not hang forever.
    """
    import httpx

    from ratiopath.openslide import OpenSlide

    slides, tiles = read_tiling(tiling_dir)
    first = slides.iloc[0]
    stem = Path(first["path"]).stem
    t = tiles[tiles["slide_id"] == first["id"]].iloc[0]

    with OpenSlide(first["path"]) as slide:
        image = slide.read_tile(int(t["x"]), int(t["y"]),
                                int(first["tile_extent_x"]), int(first["tile_extent_y"]),
                                int(first["level"]), background=255)
    print(f"probe tile: {stem} at ({t['x']},{t['y']}) level {first['level']}, "
          f"shape {image.shape} {image.dtype}")

    import asyncio

    async def _call() -> np.ndarray:
        #: follow_redirects is LOAD-BEARING here too: the router answers the bare model
        #: route with a 307 to its trailing-slash form, and httpx does not follow it by
        #: default -- without this the probe reports "dead replica" for a healthy model.
        async with httpx.AsyncClient(base_url=svc.rstrip("/"), follow_redirects=True) as client:
            # Short timeout: a dead/scaled-to-zero replica holds the connection open
            # with no response, so the readiness check must fail fast, not wait 200 s.
            return await embed_image(client, model, image, timeout=20.0)

    started = time.time()
    try:
        vec = asyncio.run(_call())
    except httpx.TimeoutException:
        sys.stderr.write(
            f"  no response within 20s -- replica is down (scaled to zero or "
            f"still UNHEALTHY). Re-run --probe once the reroll is Active.\n"
        )
        return 1
    except httpx.HTTPStatusError as e:
        sys.stderr.write(
            f"  service http {e.response.status_code}: {e.response.text[:200]!r}\n"
            "  dead replica (scaled to zero or UNHEALTHY) or bad model name.\n"
        )
        return 1
    except httpx.HTTPError as e:
        sys.stderr.write(f"  no response from {svc}: {e}\n")
        return 1
    dt = time.time() - started
    print(f"  OK: {model} returned a {vec.shape} {vec.dtype} vector in {dt:.1f}s")
    print(f"  first 5: {vec[:5].tolist()}  norm={float((vec**2).sum()**0.5):.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiling-dir", type=Path, default=TILING_PILOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--svc", default=DEFAULT_SVC,
                    help="model service base URL (no trailing slash)")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="httpx connections per actor (IBD default 512; pilot keeps it low)")
    ap.add_argument("--block-size", type=int, default=128,
                    help="rows per Ray Data block before tiling the crops")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N tiles per slide (0 = all)")
    ap.add_argument("--probe", action="store_true", help="one-tile smoke test, no batch")
    ap.add_argument("--crop-only", action="store_true",
                    help="crop tiles through the Ray pipeline but skip the model call")
    ap.add_argument("--flush-rows", type=int, default=8192,
                    help="rows per streamed batch (a finished slide is flushed as soon "
                         "as its last tile arrives)")
    args = ap.parse_args()

    if not args.tiling_dir.is_dir():
        sys.stderr.write(f"tiling pilot {args.tiling_dir} not found; "
                         "run scripts/run_tiling_pilot.py first.\n")
        return 1

    if args.probe:
        return probe(args.svc, args.model, args.tiling_dir)

    slides, tiles = read_tiling(args.tiling_dir)
    if args.limit:
        tiles = (tiles.groupby("slide_id", group_keys=False)
                 .head(args.limit).reset_index(drop=True))
        print(f"   --limit {args.limit}: {len(tiles)} tiles total")

    tiles_enriched, id_to_stem = enrich(slides, tiles)
    print(f"tiles: {len(tiles_enriched)}  slides: {slides['path'].nunique()}  "
          f"model: {args.model}  svc: {args.svc}", flush=True)

    model_dir = args.out / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    #: Per-slide durability. `ds.to_pandas()` + one `write_per_slide` at the end means a
    #: kill anywhere in a multi-day run loses every vector (that is exactly how the 40 h
    #: 186-slide run produced 0 parquets). Instead the stream is consumed batch by batch
    #: and a slide is written the moment its last tile comes back; a slide whose parquet
    #: already holds its full tile count is skipped, which is the resume path.
    counts = expected_counts(tiles, id_to_stem)
    done = already_done(model_dir, counts)
    todo = {stem: n for stem, n in counts.items() if stem not in done}
    if done:
        print(f"resume: {len(done)}/{len(counts)} slides already in {model_dir}")
    if not todo:
        print("nothing to do -- every slide already has its full parquet")
        return summary(model_dir, counts)
    print(f"to do: {sum(todo.values())} tiles over {len(todo)} slides")
    tiles_enriched = tiles_enriched[
        tiles_enriched["slide_id"].map(id_to_stem).isin(todo)
    ]

    import ray

    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    with ray.init(
        #: The pod's real CPU quota, NOT the connection count: 32 crop tasks on an 8-CPU
        #: pod starve the actors and every request ends in httpx.PoolTimeout.
        num_cpus=pod_cpu_limit(),
        #: Pin the object store: it lives in /dev/shm, whose pages count against the pod's
        #: cgroup, and Ray's OOM monitor kills off the cgroup limit. 2 GiB was too small
        #: (the driver lags the actors and the store spilled 2,329 MiB before the run
        #: died); 8 GiB leaves the 64Gi pod ample headroom and keeps blocks in memory.
        object_store_memory=8 * 1024**3,
        include_dashboard=False,
        configure_logging=False,
        runtime_env={"excludes": [".git", ".venv"]},
    ):
        ds = crop_tiles(tiles_enriched, args.block_size)
        started = time.time()

        if args.crop_only:
            n_cropped = ds.count()
            counts = (ds.drop_columns(["tile"]).to_pandas()
                      .assign(stem=lambda d: d["slide_id"].map(id_to_stem))
                      .groupby("stem").size())
            print(f"\ncropped {n_cropped} tiles in {time.time() - started:.0f}s (no model call)")
            for stem in sorted(counts.index):
                print(f"  {stem}: {int(counts[stem])} tiles cropped OK")
            return 0

        ds = add_embedding_column(ds, args.model, args.svc, args.concurrency)

        #: Only the slides still in flight are buffered, so driver memory stays at a few
        #: slides' worth of vectors rather than all 3.9M rows.
        pending: dict[str, list[pd.DataFrame]] = {}
        got: dict[str, int] = {}
        written = len(done)
        for batch in ds.iter_batches(batch_size=args.flush_rows, batch_format="pandas"):
            batch = batch.assign(stem=batch["slide_id"].map(id_to_stem))
            for stem, grp in batch.groupby("stem", sort=False):
                pending.setdefault(stem, []).append(grp)
                got[stem] = got.get(stem, 0) + len(grp)
                if got[stem] < todo[stem]:
                    continue
                flush_slide(stem, pd.concat(pending.pop(stem)), model_dir)
                written += 1
                print(f"[{written}/{len(counts)}] {stem}: {got[stem]} tiles "
                      f"({time.time() - started:.0f}s) -> {stem}.parquet", flush=True)

        #: A slide still split across the tail of the stream is embedded but unwritten;
        #: flush it so the work survives, and let the next run finish it (resume requires
        #: the full tile count, so a short parquet is redone rather than trusted).
        for stem, frames in pending.items():
            flush_slide(stem, pd.concat(frames), model_dir)
            print(f"  {stem}: flushed {got[stem]}/{todo[stem]} tiles at end of stream",
                  flush=True)

    total = time.time() - started
    print(f"\nfinished in {total:.0f}s")
    return summary(model_dir, counts)


if __name__ == "__main__":
    raise SystemExit(main())
