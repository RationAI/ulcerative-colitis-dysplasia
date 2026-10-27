"""Tile-embedding dataset for the dysplasia classifier (hunCRC).

Samples are (embedding, label, metadata):

  * embedding <- embeddings_dir/{stem}.parquet, one 1536-d row per tile
                 (prov-gigapath; generated from the tiling grid, so (x, y)
                 match the tiling exactly)
  * label     <- binary dysplasia from the TILING columns
                 (lowgrade_dysplasia | highgrade_dysplasia > 0) -- the same
                 target the U-Net experiment trained on. The label columns
                 INSIDE the embeddings parquets are stale (written by an
                 earlier tiling generation, ~1.7x more positives) and are
                 ignored.
  * split     <- slides.parquet `split` column (scripts/add_split_column.py):
                 train / test_preliminary / test_final, case-level 70/15/15.

IBD's annotated-column filter (feat/modeling preprocessing/filter_tiles.py)
is applied per slide at construction time: tiles are clustered on x-gaps and, when
the slide carries any annotation, only the annotated cluster is kept --
tiles from unannotated tissue sections are "unknown", not negative. Slides
without any annotation are kept whole (true-negative slides).

Slides are loaded lazily (an LRU of whole-slide vector arrays); no
OpenSlide is involved, so this is fork-safe with dataloader workers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _keep_annotated_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the tissue column(s) carrying annotations (IBD rule).

    Tiles are clustered by x-gaps: a gap > 5x the MEDIAN gap starts a new
    cluster. On IBD slides the between-section gap is ~35x the in-section
    stride, so sections separate cleanly; on a uniform grid (hunCRC
    single-section slides) every gap equals the median, so the whole slide
    is one cluster and the filter is a no-op. (The IBD 0.5x-max threshold
    degenerates on uniform grids: every column becomes its own cluster.)
    When the slide has any annotation, only the annotated cluster is kept.
    Raises when annotations span more than one cluster -- the IBD rule is
    one annotated section per slide, so a multi-cluster hit is a data bug
    to fail on, not silently drop.
    """
    if df["annotation"].sum() == 0:
        return df

    unique_x = np.sort(df["x"].unique())
    x_diffs = np.diff(unique_x)
    threshold = np.median(x_diffs) * 5.0 if x_diffs.size else 0.0
    clusters = np.concatenate([[0], (x_diffs > threshold).cumsum()])
    cluster_of_x = dict(zip(unique_x.tolist(), clusters.tolist()))

    out = df.copy()
    out["_cluster"] = out["x"].map(cluster_of_x)
    sums = out.groupby("_cluster")["annotation"].sum()
    valid = set(sums.index[sums > 0])
    if len(valid) > 1:
        raise RuntimeError(
            f"slide {df['slide_id'].iloc[0]}: annotations in {len(valid)} tissue "
            "clusters -- expected one annotated section per slide"
        )
    return out[out["_cluster"].isin(valid)].drop(columns=["_cluster"])


class DysplasiaEmbeddingsDataset(Dataset):
    """(embedding, binary-dysplasia label, metadata) from precomputed tiles."""

    def __init__(
        self,
        tiling_dir: str | Path,
        embeddings_dir: str | Path,
        split: str,
        max_slides: int | None = None,
        cache_size: int = 8,
    ) -> None:
        tiling_dir, embeddings_dir = Path(tiling_dir), Path(embeddings_dir)
        self.split = split
        self.cache_size = cache_size

        slides = pd.read_parquet(tiling_dir / "slides" / "slides.parquet")
        slides = slides[slides["split"] == split].copy()
        if slides.empty:
            raise SystemExit(f"no slides with split={split} in {tiling_dir}")
        slides["stem"] = slides["path"].map(lambda p: Path(p).stem)

        emb_files = {p.stem: p for p in embeddings_dir.glob("*.parquet")}
        have = slides[slides["stem"].isin(emb_files)]
        if len(have) < len(slides):
            print(
                f"[{split}] {len(slides) - len(have)}/{len(slides)} slides have no "
                f"embedding parquet in {embeddings_dir} -- using the other {len(have)}"
            )
        #: Deterministic slide order (stem-sorted); max_slides cuts from here.
        self._slides = have.sort_values("stem").reset_index(drop=True)
        if max_slides is not None:
            self._slides = self._slides.iloc[:max_slides].reset_index(drop=True)

        #: Labels come from the TILING, not the embeddings parquets (stale
        #: there). Only the columns the filter and the label need.
        tiles = pd.concat(
            [
                pd.read_parquet(
                    f,
                    columns=[
                        "slide_id",
                        "x",
                        "y",
                        "annotation",
                        "lowgrade_dysplasia",
                        "highgrade_dysplasia",
                    ],
                )
                for f in sorted(tiling_dir.glob("tiles/*.parquet"))
            ]
        )
        keep = set(self._slides["id"])
        #: reset_index: concat keeps each shard's RangeIndex, so the index
        #: labels are 4-way duplicated and groupby().groups + .iloc would
        #: select the wrong rows.
        tiles = tiles[tiles["slide_id"].isin(keep)].reset_index(drop=True)
        if tiles.empty:
            raise SystemExit(f"split {split}: no tiles for its slides")
        #: Full-grid size per slide, for the grid check in _load_slide (the
        #: annotated-column filter below legitimately shrinks the tiling side).
        self._grid_sizes = tiles.groupby("slide_id").size().to_dict()
        #: IBD's annotated-column filter, applied once per slide up front --
        #: _rows must be built AFTER it, or dropped tiles dangle as samples
        #: whose (x, y) no longer exists in the slide's position dict.
        self._tiles = pd.concat(
            (_keep_annotated_cluster(t) for _, t in tiles.groupby("slide_id")),
            ignore_index=True,
        )
        if self._tiles.empty:
            raise SystemExit(f"split {split}: no tiles left after the column filter")
        pos = (
            (self._tiles["lowgrade_dysplasia"] > 0)
            | (self._tiles["highgrade_dysplasia"] > 0)
        ).sum()
        print(
            f"[{split}] {len(self._slides)} slides, {len(self._tiles)} tiles, "
            f"{int(pos)} dysplasia ({100 * pos / len(self._tiles):.1f}%)"
        )

        self._emb_paths = {
            sid: emb_files[stem]
            for sid, stem in zip(self._slides["id"], self._slides["stem"])
        }
        #: Flat sample list (slide_id, x, y); row i of __getitem__.
        self._rows: list[tuple[str, int, int]] = list(
            zip(
                self._tiles["slide_id"].astype(str),
                self._tiles["x"].astype(int),
                self._tiles["y"].astype(int),
            )
        )
        #: Per-slide tile slice (row indices into self._tiles).
        self._tile_slice = {
            sid: idx
            for sid, idx in self._tiles.groupby("slide_id").groups.items()
        }
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def _load_slide(self, slide_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
        if slide_id in self._cache:
            return self._cache[slide_id]
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))

        tile = self._tiles.iloc[self._tile_slice[slide_id]]
        emb = pd.read_parquet(self._emb_paths[slide_id])
        #: Grid check against the FULL tile count (the annotated-column
        #: filter already shrank self._tiles at construction time).
        if len(emb) != self._grid_sizes[slide_id]:
            raise RuntimeError(
                f"slide {slide_id}: embeddings ({len(emb)} tiles) != tiling grid "
                f"({self._grid_sizes[slide_id]} tiles) -- embeddings were computed "
                "from a different tiling"
            )
        emb_pos = {
            (int(x), int(y)): i
            for i, (x, y) in enumerate(emb[["x", "y"]].itertuples(index=False, name=None))
        }
        try:
            emb_rows = np.fromiter(
                (emb_pos[(int(x), int(y))] for x, y in tile[["x", "y"]].itertuples(index=False, name=None)),
                dtype=np.int64,
                count=len(tile),
            )
        except KeyError as e:
            raise RuntimeError(
                f"slide {slide_id}: tile {e.args[0]} is in the tiling but not in the "
                "embeddings -- embeddings were computed from a different tiling"
            ) from e
        vectors = np.stack(emb["embedding"].to_list()).astype(np.float32)[emb_rows]
        labels = (
            ((tile["lowgrade_dysplasia"] > 0) | (tile["highgrade_dysplasia"] > 0))
            .to_numpy(dtype=np.float32)
        )
        position = {
            (int(x), int(y)): i
            for i, (x, y) in enumerate(tile[["x", "y"]].itertuples(index=False, name=None))
        }
        loaded = (vectors, labels, position)
        self._cache[slide_id] = loaded
        return loaded

    def __getitem__(self, idx: int):
        slide_id, x, y = self._rows[idx]
        vectors, labels, position = self._load_slide(slide_id)
        k = position[(x, y)]
        embedding = torch.from_numpy(np.ascontiguousarray(vectors[k]))
        label = torch.tensor(labels[k])
        metadata = {"slide_id": slide_id, "x": str(x), "y": str(y)}
        return embedding, label, metadata
