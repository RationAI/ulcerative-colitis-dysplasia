"""Labelled HunCRC embedding features as an on-disk ``.npy`` cache.

Builds on :mod:`ml.data.huncrc_labels`. Two passes over the per-slide embedding
parquets of the requested splits (test_final is refused):

1. cheap columns only (``slide_id, x, y, LG, HG``) -> labels -> the number of
   labelled tiles (label 0/1);
2. the ``embedding`` column in row batches, cast float64 -> float32 per batch
   and written straight into a pre-allocated ``np.lib.format.open_memmap``.

So the feature matrix never exists in RAM as float64, and never as a whole
in-memory array at all: peak RSS is one batch of float64 plus the label
columns. The cache directory holds ``features.npy`` (float32, N x D),
``labels.npy`` (int8), ``slide_names.npy`` (<U3), ``folds.npy`` (int8, -1 for
no fold), ``splits.npy``, ``x.npy``, ``y.npy``, ``coverage.npy`` and
``meta.json``. It is keyed by the label config (``min_coverage`` first), the
splits and the slide list; a different key builds a new directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ml.data import huncrc_labels
from ml.data.huncrc_labels import (
    LABEL_EXCLUDED,
    LabelConfig,
    assign_labels,
    load_tiles,
    split_slide_ids,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd


#: Bump when the cache LAYOUT or the feature extraction changes.
CACHE_VERSION = 2
EMBEDDING = "embedding"


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray  # float32 (N, D), memory-mapped by default
    labels: np.ndarray  # int8, 0/1
    slide_names: np.ndarray
    folds: np.ndarray  # int8, -1 = no fold (test splits)
    splits: np.ndarray
    x: np.ndarray
    y: np.ndarray
    coverage: np.ndarray
    meta: dict[str, Any]

    def __len__(self) -> int:
        return len(self.labels)

    def select(self, mask: np.ndarray) -> FeatureSet:
        """Row subset. Features stay a view only for slices; fancy masks copy."""
        idx = np.flatnonzero(mask)
        return FeatureSet(
            features=self.features[idx],
            labels=self.labels[idx],
            slide_names=self.slide_names[idx],
            folds=self.folds[idx],
            splits=self.splits[idx],
            x=self.x[idx],
            y=self.y[idx],
            coverage=self.coverage[idx],
            meta=self.meta,
        )


_ARRAYS = ("labels", "slide_names", "folds", "splits", "x", "y", "coverage")


class StaleCacheError(RuntimeError):
    """The cache was built under different label rules or a different cache version."""


def _config_dict(config: LabelConfig) -> dict[str, Any]:
    return {
        k: sorted(v) if isinstance(v, frozenset) else v
        for k, v in asdict(config).items()
    }


def cache_key(
    config: LabelConfig, splits: Iterable[str], slide_names: Iterable[str]
) -> str:
    """``mc<min_coverage>-<splits>-r<rule fingerprint>-v<cache version>-<slides hash>``.

    The rule fingerprint covers min_coverage, exclude_slides, safe_categories,
    exclude_serrated and ``LABEL_RULES_VERSION``; changing any of them, or
    ``CACHE_VERSION``, gives a new key and therefore a new build.
    """
    splits = sorted(splits)
    rules = huncrc_labels.rules_fingerprint(config)
    payload = {"splits": splits, "slides": sorted(slide_names)}
    blob = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(blob).hexdigest()[:8]
    return (
        f"mc{config.min_coverage:g}-{'+'.join(splits)}"
        f"-r{rules}-v{CACHE_VERSION}-{digest}"
    )


def check_cache_meta(meta: dict[str, Any], config: LabelConfig | None) -> None:
    """Raise :class:`StaleCacheError` if ``meta`` does not match the current code."""
    if meta.get("version") != CACHE_VERSION:
        raise StaleCacheError(
            f"cache version {meta.get('version')} != {CACHE_VERSION}; rebuild"
        )
    if meta.get("label_rules_version") != huncrc_labels.LABEL_RULES_VERSION:
        raise StaleCacheError(
            f"label rules version {meta.get('label_rules_version')} != "
            f"{huncrc_labels.LABEL_RULES_VERSION}; rebuild"
        )
    if config is not None:
        want = huncrc_labels.rules_fingerprint(config)
        if meta.get("rules_fingerprint") != want:
            raise StaleCacheError(
                f"cache built with rules {meta.get('rules_fingerprint')} "
                f"({meta.get('config')}), expected {want} ({_config_dict(config)})"
            )


def _embedding_block(column: pa.Array | pa.ChunkedArray, dim: int) -> np.ndarray:
    """``list<double>`` column -> float64 (n, dim) array, validating every row."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if column.null_count:
        raise ValueError("null embeddings")
    lengths = pc.list_value_length(column).to_numpy(zero_copy_only=False)
    if len(lengths) and (lengths != dim).any():
        raise ValueError(f"embedding length {sorted(set(lengths.tolist()))} != {dim}")
    flat = column.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(len(column), dim)


def build_features(
    emb_dir: Path,
    slide_table: pd.DataFrame,
    splits: Iterable[str],
    config: LabelConfig,
    cache_root: Path,
    dim: int = 1536,
    batch_size: int = 8192,
    overwrite: bool = False,
) -> Path:
    """Build (or reuse) the cache for ``splits``; returns its directory."""
    splits = tuple(sorted(set(splits)))
    wanted = split_slide_ids(slide_table, splits)  # refuses test_final
    key = cache_key(config, splits, wanted["name"])
    out = Path(cache_root) / key
    if (out / "meta.json").is_file() and not overwrite:
        check_cache_meta(json.loads((out / "meta.json").read_text()), config)
        return out

    # pass 1: labels from cheap columns, per-slide files of `splits` only
    tiles = load_tiles(emb_dir, slide_table, splits)
    tiles = assign_labels(tiles, slide_table, config)
    keep = (tiles["label"] != LABEL_EXCLUDED).to_numpy()
    kept = tiles.loc[keep]
    n = int(keep.sum())

    tmp = Path(cache_root) / f".tmp-{key}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    feats = np.lib.format.open_memmap(
        tmp / "features.npy", mode="w+", dtype=np.float32, shape=(n, dim)
    )

    # pass 2: embeddings, same files in the same order, batch by batch
    pos = 0
    id_to_name = dict(zip(wanted["id"], wanted["name"], strict=True))
    for slide_id, name in id_to_name.items():
        slide_rows = tiles["slide_id"].to_numpy() == slide_id
        slide_keep = keep[slide_rows]
        slide_xy = tiles.loc[slide_rows, ["x", "y"]].to_numpy()
        pf = pq.ParquetFile(Path(emb_dir) / f"{name}.parquet")
        offset = 0
        for batch in pf.iter_batches(
            batch_size=batch_size, columns=["slide_id", "x", "y", EMBEDDING]
        ):
            # per-slide file: every row must be this slide (load_tiles asserted the same)
            sid = batch.column("slide_id").to_numpy(zero_copy_only=False)
            if (sid != slide_id).any():
                raise ValueError(f"{name}.parquet contains rows of other slides")
            m = len(sid)
            k = slide_keep[offset : offset + m]
            xy = np.stack(
                [batch.column("x").to_numpy(), batch.column("y").to_numpy()], axis=1
            )
            if not np.array_equal(xy, slide_xy[offset : offset + m]):
                raise ValueError(f"{name}.parquet: row order differs between passes")
            if k.any():
                block = _embedding_block(batch.column(EMBEDDING), dim)[k].astype(
                    np.float32
                )
                if not np.isfinite(block).all():
                    raise ValueError(f"{name}.parquet: non-finite embeddings")
                feats[pos : pos + len(block)] = block
                pos += len(block)
                del block
            offset += m
        if offset != len(slide_keep):
            raise ValueError(
                f"{name}.parquet: {offset} rows, pass 1 saw {len(slide_keep)}"
            )
    assert pos == n, (pos, n)
    feats.flush()
    del feats

    folds = kept["fold"].fillna(-1).astype(np.int8).to_numpy()
    arrays = {
        "labels": kept["label"].astype(np.int8).to_numpy(),
        "slide_names": kept["name"].to_numpy().astype("U3"),
        "folds": folds,
        "splits": kept["split"].to_numpy().astype("U16"),
        "x": kept["x"].to_numpy(np.int64),
        "y": kept["y"].to_numpy(np.int64),
        "coverage": kept["coverage"].to_numpy(np.float32),
    }
    for k, v in arrays.items():
        np.save(tmp / f"{k}.npy", v)
    meta = {
        "version": CACHE_VERSION,
        "label_rules_version": huncrc_labels.LABEL_RULES_VERSION,
        "rules_fingerprint": huncrc_labels.rules_fingerprint(config),
        "key": key,
        "config": _config_dict(config),
        "splits": list(splits),
        "slides": sorted(wanted["name"]),
        "n": n,
        "dim": dim,
        "n_pos": int((arrays["labels"] == 1).sum()),
        "n_neg": int((arrays["labels"] == 0).sum()),
        "source": str(emb_dir),
        "split_fold_source": "slides.parquet",
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def load_features(
    cache_dir: Path, expected_config: LabelConfig | None = None, mmap: bool = True
) -> FeatureSet:
    """Load a cache; with ``expected_config`` its label rules are verified.

    The cache and label-rule versions are always checked against the code.
    """
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    check_cache_meta(meta, expected_config)
    features = np.load(cache_dir / "features.npy", mmap_mode="r" if mmap else None)
    arrays = {k: np.load(cache_dir / f"{k}.npy") for k in _ARRAYS}
    if features.dtype != np.float32 or features.shape != (meta["n"], meta["dim"]):
        raise ValueError(
            f"corrupt cache {cache_dir}: {features.dtype} {features.shape}"
        )
    if any(len(a) != meta["n"] for a in arrays.values()):
        raise ValueError(f"corrupt cache {cache_dir}: length mismatch")
    if "test_final" in set(arrays["splits"]):
        raise PermissionError("cache contains test_final rows")
    return FeatureSet(features=features, meta=meta, **arrays)


def get_features(
    emb_dir: Path,
    slide_table: pd.DataFrame,
    splits: Iterable[str],
    config: LabelConfig,
    cache_root: Path,
    **kwargs: Any,
) -> FeatureSet:
    return load_features(
        build_features(emb_dir, slide_table, splits, config, cache_root, **kwargs),
        expected_config=config,
    )
