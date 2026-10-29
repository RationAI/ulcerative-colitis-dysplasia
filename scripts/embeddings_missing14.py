#!/usr/bin/env python
"""Embed the 14 hunCRC slides missing from preprocessing/embeddings/prov-gigapath.

Share-based twin of preprocessing/embeddings.py's job path, for the
tiling_missing14 output: reads the 14 slides' tiling parquets straight from the
share (no MLflow URIs), crops every tile from the whole slides and POSTs it to
the prov-gigapath model service, and writes one parquet per slide into the
EXISTING embeddings dir -- new files only, none of the 186 existing parquets is
touched. The output columns match the existing parquets exactly (the tiling's
`split` column is dropped, `stem` is added on write), so the training
consumer (ml/data/embeddings_dataset.py) globs the dir and finds 200 uniform
files.

Row-count contract the consumer enforces (_read_vectors): each {stem}.parquet
must carry exactly the tiling's per-slide tile count or training fails to start.
It holds by construction -- every tiling tile of the 14 is embedded -- and is
re-checked against the tiling before and after the write.

Usage:
    .venv/bin/python scripts/embeddings_missing14.py             # all 14 (job mode)
    .venv/bin/python scripts/embeddings_missing14.py --probe-only  # service check
    .venv/bin/python scripts/embeddings_missing14.py --smoke       # 64 tiles -> /tmp
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
#: sys.path[0] is scripts/, where `preprocessing/` holds the JOB SUBMITTERS --
#: the repo root (the real `preprocessing` package) must win.
sys.path.insert(0, str(REPO_ROOT))

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
TILING = PROJECT / "preprocessing" / "tiling_missing14"
OUT = PROJECT / "preprocessing" / "embeddings" / "prov-gigapath"

#: Same model/service/concurrency as configs/preprocessing/embeddings.yaml.
MODEL = "prov-gigapath"
SVC = (
    "http://rayservice-model-huncrc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000"
)
CONCURRENCY = 512
BLOCK_SIZE = 128
#: The tiling column the 186 parquets do not carry (they predate the split
#: stamp) -- drop it so the 14 match the dir's existing schema exactly.
DROP_COLS = ["split"]


def tiling_stems() -> list[str]:
    stems = sorted(p.stem for p in (TILING / "tiles").glob("*.parquet"))
    if not stems:
        sys.exit(f"no tile parquets in {TILING / 'tiles'}")
    return stems


def probe_service(svc: str) -> None:
    """One white-tile POST -- the same wire contract as EmbedTiles. A dead or
    redeployed service fails here in seconds instead of mid-job."""
    import httpx

    from preprocessing.embeddings import embed_image

    async def _probe() -> None:
        async with httpx.AsyncClient(
            base_url=svc.rstrip("/"),
            follow_redirects=True,  # load-bearing, see EmbedTiles
            timeout=60.0,
        ) as client:
            tile = np.full((224, 224, 3), 255, dtype=np.uint8)
            vec = await embed_image(client, MODEL, tile)
        print(f"service probe ok: {vec.shape[0]}-d vector")

    asyncio.run(_probe())


def write_per_slide_clean(
    df: pd.DataFrame, id_to_stem: dict[str, str], out_dir: Path
) -> None:
    """preprocessing.embeddings.write_per_slide with the index reset inside the
    loop: on pandas 2.3.3 a groupby subframe carries a plain Index (not a
    RangeIndex) and to_parquet stores it as a __index_level_0__ column -- none of
    the existing 186 parquets has one, and the 14 must match the dir's schema."""
    df = df.assign(stem=df["slide_id"].map(id_to_stem))
    for stem, grp in df.groupby("stem"):
        grp.reset_index(drop=True).to_parquet(out_dir / f"{stem}.parquet")


def load_tiling() -> tuple[pd.DataFrame, pd.DataFrame]:
    slides = pd.concat(
        (pd.read_parquet(f) for f in sorted((TILING / "slides").glob("*.parquet"))),
        ignore_index=True,
    )
    tiles = pd.concat(
        (pd.read_parquet(f) for f in sorted((TILING / "tiles").glob("*.parquet"))),
        ignore_index=True,
    )
    missing = set(slides["id"]) - set(tiles["slide_id"])
    if missing:
        sys.exit(f"slides without tiles: {missing}")
    return slides, tiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-only", action="store_true",
                    help="check the model service and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="embed 64 tiles of the first slide to /tmp (no share write)")
    ap.add_argument("--svc", default=SVC,
                    help="model service base URL (default: the huncrc RayService)")
    args = ap.parse_args()

    svc = args.svc
    stems = tiling_stems()
    print(f"{len(stems)} slides to embed: {', '.join(stems)}")

    probe_service(svc)
    if args.probe_only:
        return

    slides, tiles = load_tiling()

    existing = [s for s in stems if (OUT / f"{s}.parquet").exists()]
    if existing and not args.smoke:
        sys.exit(
            f"already embedded in {OUT}: {existing} -- removing them first would "
            "overwrite existing data, which this script refuses to do"
        )

    smoke = args.smoke
    if smoke:
        stem0 = stems[0]
        keep = slides["path"].map(lambda p: Path(p).stem) == stem0
        slides = slides[keep]
        sid0 = slides["id"].iloc[0]
        tiles = tiles[tiles["slide_id"] == sid0].head(64)
        out_dir = Path("/tmp/embeddings_smoke")
    else:
        out_dir = OUT

    #: Same enrichment as preprocessing/embeddings.py main().
    slide_info = slides.set_index("id")[
        ["path", "level", "tile_extent_x", "tile_extent_y"]
    ]
    tiles_enriched = tiles.join(slide_info, on="slide_id")
    if tiles_enriched[["path", "level"]].isna().any().any():
        sys.exit("unmatched slide_ids in the tiles join")
    for c in DROP_COLS:
        if c in tiles_enriched.columns:
            tiles_enriched = tiles_enriched.drop(columns=c)

    import ray

    from preprocessing.embeddings import (
        add_embedding_column,
        crop_tiles,
    )

    ray.init(runtime_env={"excludes": [".git", ".venv"]})
    try:
        ds = crop_tiles(tiles_enriched, BLOCK_SIZE)
        ds = add_embedding_column(ds, MODEL, svc, CONCURRENCY)
        df = ds.to_pandas()
    finally:
        ray.shutdown()

    #: The consumer's row-count contract, checked before the write.
    want = tiles.groupby("slide_id").size()
    got = df.groupby("slide_id").size()
    if not want.equals(got):
        sys.exit(f"row counts differ from the tiling:\nwant\n{want}\ngot\n{got}")

    id_to_stem = dict(
        zip(slides["id"], slides["path"].map(lambda p: Path(p).stem))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_slide_clean(df, id_to_stem, out_dir)

    sid_of = dict(zip(id_to_stem.values(), id_to_stem))
    for s in (stems[:1] if smoke else stems):
        n = len(pd.read_parquet(out_dir / f"{s}.parquet"))
        print(f"{s}: {n} tiles (tiling: {int(want[sid_of[s]])})")
    if not smoke:
        total = len(list(OUT.glob("*.parquet")))
        print(f"\n{OUT} now holds {total} slide parquets")
        print("next: point embeddings_dir at preprocessing/embeddings/prov-gigapath")


if __name__ == "__main__":
    main()
