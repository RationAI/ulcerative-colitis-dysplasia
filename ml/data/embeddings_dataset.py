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

Labeling convention: by default every tile is a training sample (annotated
= positive, unannotated tissue = negative -- the same convention the U-Net
experiment uses). With ``cluster_k`` set, the serial-sections rule is
applied instead: tiles are grouped into 2D-connected tissue pieces
(connected = within ``cluster_k * stride`` of a neighbour), and on any slide
carrying an annotation only the annotated pieces are kept -- the other
pieces are consecutive sections cut from the same region (the pathologist
annotated one and skipped the rest), so they are unknown, not negative.
Slides without any annotation are always kept whole (true-negative slides).

All vectors of the split are read once, in __init__, into a single contiguous
float32 array. The train sampler shuffles over ~900k tiles, so slides are
visited in random order and any per-slide cache misses almost every time --
a lazy loader re-reads a ~130 MB parquet (plus a 21k-row grid check) per
batch. The array is read-only after construction, so the forked dataloader
workers share it copy-on-write instead of each holding its own copy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset


def _annotated_piece_mask(
    df: pd.DataFrame, stride: float, cluster_k: float
) -> np.ndarray:
    """Boolean keep-mask: tiles inside 2D-connected pieces that carry annotations.

    A piece is a connected component of tiles whose neighbours are within
    ``cluster_k * stride`` (Euclidean). At ``cluster_k=2`` a piece stays
    connected across a single missing-tile hole (gap ``2*stride``) while two
    separate fragments -- observed minimum gap in the cohort: ``2.236*stride``
    -- never merge. Slides without any annotation return all-True
    (true-negative slides are kept whole).
    """
    n = len(df)
    ann = (df["annotation"] > 0).to_numpy()
    if not ann.any():
        return np.ones(n, dtype=bool)

    pts = np.column_stack([df["x"].to_numpy(), df["y"].to_numpy()]).astype(float)
    pairs = np.asarray(
        list(cKDTree(pts).query_pairs(r=cluster_k * stride)), dtype=np.int64
    )
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, comp = np.unique(roots, return_inverse=True)
    has_ann = np.zeros(int(comp.max()) + 1, dtype=bool)
    np.logical_or.at(has_ann, comp[ann], True)
    return has_ann[comp]


class DysplasiaEmbeddingsDataset(Dataset):
    """(embedding, binary-dysplasia label, metadata) from precomputed tiles."""

    def __init__(
        self,
        tiling_dir: str | Path,
        embeddings_dir: str | Path,
        split: str,
        max_slides: int | None = None,
        cluster_k: float | None = None,
    ) -> None:
        tiling_dir, embeddings_dir = Path(tiling_dir), Path(embeddings_dir)
        self.split = split
        self.cluster_k = cluster_k

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
        #: there). Only the columns the label (and the cluster filter) needs.
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
        #: Full (pre-filter) per-slide grid size -- _read_vectors checks the
        #: embeddings grid against this, since the filter below may shrink
        #: the tile side.
        self._grid_sizes = tiles.groupby("slide_id").size().to_dict()
        if self.cluster_k is not None:
            stride_of = dict(
                zip(
                    self._slides["id"],
                    np.maximum(
                        self._slides["stride_x"], self._slides["stride_y"]
                    ).to_numpy(dtype=float),
                )
            )
            stem_of = dict(zip(self._slides["id"], self._slides["stem"]))
            parts = []
            for sid, t in tiles.groupby("slide_id", sort=False):
                mask = _annotated_piece_mask(t, stride_of[sid], self.cluster_k)
                kept = t[mask]
                if len(kept) < len(t):
                    print(
                        f"[{split}] {stem_of[sid]}: kept {len(kept)}/{len(t)} tiles "
                        f"(annotated pieces only)"
                    )
                parts.append(kept)
            tiles = pd.concat(parts, ignore_index=True)
        self._tiles = tiles
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
        #: Row indices of each slide inside self._tiles.
        self._tile_slice = {
            sid: np.asarray(idx)
            for sid, idx in self._tiles.groupby("slide_id").groups.items()
        }
        self._labels = (
            (
                (self._tiles["lowgrade_dysplasia"] > 0)
                | (self._tiles["highgrade_dysplasia"] > 0)
            )
            .to_numpy(dtype=np.float32)
        )
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._read_vectors()

    def _read_vectors(self) -> None:
        """Read every slide's vectors into one (n_tiles, dim) float32 array."""
        self._vectors = None
        n = len(self._emb_paths)
        for done, (sid, path) in enumerate(self._emb_paths.items(), 1):
            idx = self._tile_slice[sid]
            tile = self._tiles.iloc[idx]
            emb = pd.read_parquet(path)
            #: Grid check against the FULL (pre-filter) tiling count -- the
            #: cluster filter legitimately shrinks the tile side.
            if len(emb) != self._grid_sizes[sid]:
                raise RuntimeError(
                    f"slide {sid}: embeddings ({len(emb)} tiles) != tiling grid "
                    f"({self._grid_sizes[sid]} tiles) -- embeddings were computed "
                    "from a different tiling"
                )
            emb_pos = {
                (int(x), int(y)): k
                for k, (x, y) in enumerate(
                    emb[["x", "y"]].itertuples(index=False, name=None)
                )
            }
            try:
                emb_rows = np.fromiter(
                    (
                        emb_pos[(int(x), int(y))]
                        for x, y in tile[["x", "y"]].itertuples(
                            index=False, name=None
                        )
                    ),
                    dtype=np.int64,
                    count=len(tile),
                )
            except KeyError as e:
                raise RuntimeError(
                    f"slide {sid}: tile {e.args[0]} is in the tiling but not in the "
                    "embeddings -- embeddings were computed from a different tiling"
                ) from e
            vectors = np.stack(emb["embedding"].to_list()).astype(np.float32)
            if self._vectors is None:
                self._vectors = np.empty(
                    (len(self._tiles), vectors.shape[1]), dtype=np.float32
                )
            self._vectors[idx] = vectors[emb_rows]
            del emb, vectors
            print(
                f"[{self.split}] slide {done}/{n} {sid}: "
                f"{self._vectors.nbytes / 2**30:.1f} GiB held",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        slide_id, x, y = self._rows[idx]
        embedding = torch.from_numpy(np.ascontiguousarray(self._vectors[idx]))
        label = torch.tensor(self._labels[idx])
        metadata = {"slide_id": slide_id, "x": str(x), "y": str(y)}
        return embedding, label, metadata
