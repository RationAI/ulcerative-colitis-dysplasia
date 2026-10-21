"""Tile-embedding stage: crop each tiled region and run it through the embedding model.

    # job path -- reads HunCRC tiling from MLflow URIs and uploads the vectors
    uv run python -m preprocessing.embeddings +preprocessing=embeddings

Ported from the IBD `feature/embeddings:preprocessing/embeddings.py`. That script reads
`config.dataset.mlflow_uris.tiling_filtered` (IBD experiment-111 artifacts, dead for
HunCRC) and calls `AsyncClient.models.embed_image`. Three HunCRC deltas, each forced by
the environment rather than a design choice:

1. SELF-CONTAINED embed_image. The rationai-sdk pinned in this env
   (gitlab a4d2508, rationai-sdk-0.1.0) has NO `models.embed_image` -- only
   classify_image/segment_image. The IBD branch was written against the newer GitHub
   RationAI/rationai-sdk-python@3fb74c0, which added it. Rather than bump the SDK and
   churn pyproject/uv.lock, the call is inlined below as httpx + lz4, matching the real
   wire contract the newer SDK documents (POST <svc>/<model>, body
   lz4.frame.compress(image.tobytes()), request header `x-output-dtype`, response
   lz4.frame.decompress(content) reshaped by the `x-output-shape` response header).

2. MODEL + SERVICE FROM CONFIG. IBD hardcodes `rayservice-model-fix-serve-svc`; here
   `model` and `svc` come from the embeddings config, so the same stage can target the
   HunCRC rerolled deployment.

3. OUTPUT LAYOUT THE CONSUMER ACTUALLY READS. IBD wrote `{output_dir}/{split}/tiles/*.parquet`
   (output_dir = ${project_path}/embeddings/${mpp}mpp), but the training dataset
   (feat/modeling `DysplasiaEmbeddingsDataset`) globs `embeddings_dir/*.parquet` where
   `embeddings_dir = ${project_path}/embeddings/${embedd_model}`, matching each slide_id
   to a file by its filename stem (dataset.py:44-73) and reading x, y, embedding.
   The two IBD layouts never matched; this stage writes the one the consumer reads:
   one parquet per slide, named {stem}.parquet, directly under {output_dir}/{model}/.
   Slide stems are unique across splits, so flattening splits into one dir is unambiguous.
"""

from __future__ import annotations

import lz4.frame
from pathlib import Path
from typing import Any

import hydra
import httpx
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col


#: A 224px slide tile is ~150 KB raw uint8 RGB; this bounds a hung replica, not a slow
#: slide (each call is one small crop, not a per-slide job). IBD's per-call timeout.
CALL_TIMEOUT = 200.0


async def embed_image(
    client: httpx.AsyncClient, model: str, image: np.ndarray,
    timeout: float = CALL_TIMEOUT,
) -> np.ndarray:
    """One tile -> embedding vector, mirroring the SDK's embed_image wire contract.

    `image` is a uint8 RGB (H, W, 3) array from read_slide_tiles (background filled
    white). The vector is decoded as float32 and reshaped by the x-output-shape header.
    """
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    data = lz4.frame.compress(arr.tobytes())
    headers = {"x-output-dtype": "float32"}
    r = await client.post(model, data=data, headers=headers, timeout=timeout)
    r.raise_for_status()
    buf = lz4.frame.decompress(r.content)
    vec = np.frombuffer(buf, dtype=np.float32)
    #: e.g. "x-output-shape: (2560,)" -> (2560,). The SDK evals the header verbatim.
    shape = eval(r.headers["x-output-shape"])  # noqa: S307 -- SDK parity
    return vec.reshape(shape)


class EmbedTiles:
    """Ray actor holding one httpx.AsyncClient; maps one tile row -> row + embedding."""

    def __init__(self, model: str, svc: str, concurrency: int) -> None:
        #: follow_redirects is LOAD-BEARING: the Ray Serve router answers the
        #: model's bare route with a 307 to the trailing-slash form, and httpx
        #: defaults to NOT following it -- so every tile would raise 307 and the
        #: whole run fails, even though the model is up. The single-tile probe
        #: passes with this flag; without it the batch does not.
        self.client = httpx.AsyncClient(
            base_url=svc.rstrip("/"),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=concurrency,
                                max_keepalive_connections=concurrency),
            timeout=httpx.Timeout(CALL_TIMEOUT),
        )
        self.model = model

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        vec = await embed_image(self.client, self.model, row["tile"])
        del row["tile"]
        row["embedding"] = vec.tolist()
        return row


def crop_tiles(
    tiles_enriched: pd.DataFrame, block_size: int
) -> ray.data.Dataset:
    """The tiling-geometry -> in-memory uint8-RGB-tile half, shared with the pilot.

    Joins are assumed done by the caller (tiles carry path, x, y, tile_extent, level).
    """
    return (
        ray.data.from_arrow(pa.Table.from_pandas(tiles_enriched, preserve_index=False))
        .repartition(target_num_rows_per_block=block_size)
        .with_column(
            "tile",
            read_slide_tiles(  # pyright: ignore[reportCallIssue]
                col("path"), col("x"), col("y"),
                col("tile_extent_x"), col("tile_extent_y"), col("level"),
            ),
            num_cpus=1,
            memory=4 * 1024**3,
        )
        .drop_columns(["path", "level", "tile_extent_x", "tile_extent_y"])
    )


def add_embedding_column(
    ds: ray.data.Dataset, model: str, svc: str, concurrency: int
) -> ray.data.Dataset:
    """The crop -> embedding half, shared with the pilot."""
    return ds.map(
        EmbedTiles,  # pyright: ignore[reportArgumentType]
        fn_constructor_args=(model, svc, concurrency),
        compute=ray.data.ActorPoolStrategy(
            max_size=4,
            max_tasks_in_flight_per_actor=max(concurrency // 4, 1),
        ),
        max_concurrency=concurrency,
    )


def write_per_slide(df: pd.DataFrame, id_to_stem: dict[str, str], out_dir: Path) -> None:
    """One parquet per slide, named {stem}.parquet, so the training consumer's
    glob("*.parquet") + stem match finds it. Keeps x, y, embedding + label columns."""
    df = df.assign(stem=df["slide_id"].map(id_to_stem))
    for stem, grp in df.groupby("stem"):
        grp.to_parquet(out_dir / f"{stem}.parquet")


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    model = str(config.model)
    svc = str(config.svc)
    out_dir = Path(config.output_dir) / model
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_uri in config.dataset.mlflow_uris.tiling_filtered.items():
        folder = Path(mlflow.artifacts.download_artifacts(split_uri))
        slides = pd.read_parquet(folder / "slides")
        tiles = pd.read_parquet(folder / "tiles")

        #: slides -> (path, level, tile_extent) exactly as IBD's slide_info.
        slide_info = slides.set_index("id")[["path", "level", "tile_extent_x", "tile_extent_y"]]
        tiles_enriched = tiles.join(slide_info, on="slide_id")
        id_to_stem = dict(zip(slides["id"], [Path(p).stem for p in slides["path"]]))

        ds = crop_tiles(tiles_enriched, int(config.block_size))
        ds = add_embedding_column(ds, model, svc, int(config.concurrency))

        df = ds.to_pandas()
        write_per_slide(df, id_to_stem, out_dir)
        print(f"  {name}: {len(df)} tiles -> {out_dir}/", flush=True)

    logger.log_artifacts(config.output_dir, config.mlflow_artifact_path)


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
