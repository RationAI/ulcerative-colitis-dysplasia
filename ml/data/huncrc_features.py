"""Labelled HunCRC embedding features as an on-disk ``.npy`` cache.

Builds on :mod:`ml.data.huncrc_labels`. Two passes over the per-slide embedding
parquets of the requested splits (test_final is refused):

1. cheap columns only (``slide_id, x, y, LG, HG``) -> labels -> the number of
   labelled tiles (label 0/1);
2. the ``embedding`` column in row batches, cast to float32 per batch (stored
   as float64, float32 or float16: prov-gigapath files hold float64, Virchow2
   files float16)
   and written straight into a pre-allocated ``np.lib.format.open_memmap``.

So the feature matrix never exists in RAM as float64, and never as a whole
in-memory array at all: peak RSS is one batch of float64 plus the label
columns. The cache directory holds ``features.npy`` (float32, N x D),
``labels.npy`` (int8), ``slide_names.npy`` (<U3), ``folds.npy`` (int8, -1 for
no fold), ``splits.npy``, ``x.npy``, ``y.npy``, ``coverage.npy`` and
``meta.json``. It is keyed by the label config (``min_coverage`` first), the
splits, the slide list and the embedding (model + dimension); a different key
builds a new directory. Caches of the default embedding (prov-gigapath,
1536-d) keep the keys they had before the embedding entered the key; any other
model or dimension adds ``-e<model>-d<dim>``, so two embeddings can never share
a cache. Loading can verify the model and dimension (``check_cache_meta``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ml.data import huncrc_labels
from ml.data.huncrc_annotations import load_squares, tile_flags
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
#: The embedding every cache was built from before the model entered the key.
#: Its keys carry no embedding tag, so those caches stay valid.
DEFAULT_MODEL = "prov-gigapath"
DEFAULT_DIM = 1536
_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_FLOAT_TYPES = (pa.float16(), pa.float32(), pa.float64())


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
    #: only in "all"-tile caches: exclusion reason ("" = labelled), LG, HG
    reasons: np.ndarray | None = None
    lg: np.ndarray | None = None
    hg: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def embedding(self) -> tuple[str, int]:
        """(model, dim) this cache was built from."""
        return cache_embedding(self.meta)

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
            reasons=None if self.reasons is None else self.reasons[idx],
            lg=None if self.lg is None else self.lg[idx],
            hg=None if self.hg is None else self.hg[idx],
        )


_ARRAYS = ("labels", "slide_names", "folds", "splits", "x", "y", "coverage")
_ALL_TILE_ARRAYS = ("reasons", "lg", "hg")
TILE_MODES = ("labelled", "all")


class StaleCacheError(RuntimeError):
    """The cache was built under different label rules or a different cache version."""


class EmbeddingMismatchError(RuntimeError):
    """The cache holds a different embedding model or dimension than expected."""


def embedding_tag(model: str, dim: int) -> str:
    """Cache-key part for the embedding; empty for the default (prov-gigapath, 1536)."""
    if not _MODEL_NAME.match(model):
        raise ValueError(
            f"embedding model must match {_MODEL_NAME.pattern}, got {model!r}"
        )
    if int(dim) != dim or dim <= 0:
        raise ValueError(f"embedding dimension must be a positive int, got {dim!r}")
    return "" if (model, dim) == (DEFAULT_MODEL, DEFAULT_DIM) else f"-e{model}-d{dim}"


def cache_embedding(meta: dict[str, Any]) -> tuple[str, int]:
    """(model, dim) of a cache.

    Caches built before ``embedding_model`` was recorded name it by their
    source dir (``.../embeddings/prov-gigapath``).
    """
    model = meta.get("embedding_model") or Path(str(meta["source"])).name
    return str(model), int(meta["dim"])


def _config_dict(config: LabelConfig) -> dict[str, Any]:
    return {
        k: sorted(v) if isinstance(v, frozenset) else v
        for k, v in asdict(config).items()
    }


def cache_key(
    config: LabelConfig,
    splits: Iterable[str],
    slide_names: Iterable[str],
    mode: str = "labelled",
    *,
    model: str = DEFAULT_MODEL,
    dim: int = DEFAULT_DIM,
) -> str:
    """``mc<min_coverage>-<splits>-r<rules>[-e<model>-d<dim>]-v<version>-<slides hash>``.

    The rule fingerprint covers min_coverage, exclude_slides, safe_categories,
    exclude_serrated and ``LABEL_RULES_VERSION``; changing any of them, or
    ``CACHE_VERSION``, gives a new key and therefore a new build. An
    ``mode="all"`` cache (every tile, for evaluation only) gets ``-all`` after
    the splits; labelled-cache keys are unchanged. The embedding tag is empty
    for the default embedding (see :func:`embedding_tag`).
    """
    if mode not in TILE_MODES:
        raise ValueError(f"mode must be one of {TILE_MODES}, got {mode!r}")
    splits = sorted(splits)
    rules = huncrc_labels.rules_fingerprint(config)
    payload = {"splits": splits, "slides": sorted(slide_names)}
    blob = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(blob).hexdigest()[:8]
    return (
        f"mc{config.min_coverage:g}-{'+'.join(splits)}"
        f"{'-all' if mode == 'all' else ''}"
        f"-r{rules}{embedding_tag(model, dim)}-v{CACHE_VERSION}-{digest}"
    )


def check_cache_meta(
    meta: dict[str, Any],
    config: LabelConfig | None,
    model: str | None = None,
    dim: int | None = None,
) -> None:
    """Refuse a cache that does not match the code or the expected embedding.

    :class:`StaleCacheError` for other label rules or versions,
    :class:`EmbeddingMismatchError` for another embedding than ``model`` /
    ``dim`` (each checked only when given).
    """
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
    got_model, got_dim = cache_embedding(meta)
    if (model is not None and got_model != model) or (
        dim is not None and got_dim != dim
    ):
        raise EmbeddingMismatchError(
            f"REFUSED: cache {meta.get('key')} holds {got_model} {got_dim}-d embeddings "
            f"(source {meta.get('source')}), expected "
            f"{model or got_model} {dim or got_dim}-d. Embeddings of different models "
            "are never mixed; build a cache for this embedding."
        )


def _check_embedding_dim(path: Path, dim: int) -> None:
    """Refuse before building when the first stored vector is not ``dim`` long."""
    pf = pq.ParquetFile(path)
    first = next(pf.iter_batches(batch_size=1, columns=[EMBEDDING]), None)
    if first is None or first.num_rows == 0:
        return  # an empty slide file; pass 2 checks every row anyway
    got = len(first.column(EMBEDDING)[0].as_py())
    if got != dim:
        raise ValueError(
            f"embedding length {got} in {path.name} != dim {dim}: pass the "
            "dimension of this embedding model"
        )


def _embedding_block(column: pa.Array | pa.ChunkedArray, dim: int) -> np.ndarray:
    """``list<float16|float32|float64>`` column -> (n, dim) array, stored dtype.

    Every row is validated; the caller casts to float32.
    """
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    value_type = getattr(column.type, "value_type", None)
    if value_type not in _FLOAT_TYPES:
        raise TypeError(
            f"embedding column must be a list of float16/float32/float64, got {column.type}"
        )
    if column.null_count:
        raise ValueError("null embeddings")
    lengths = pc.list_value_length(column).to_numpy(zero_copy_only=False)
    if len(lengths) and (lengths != dim).any():
        raise ValueError(f"embedding length {sorted(set(lengths.tolist()))} != {dim}")
    flat = column.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(len(column), dim)


def attach_annotation_flags(tiles: pd.DataFrame, annotations: Path) -> pd.DataFrame:
    """Add normal / strict_normal / carcinoma / carcinoma_any (bool) per tile."""
    out = tiles.copy()
    cols = ("normal", "strict_normal", "carcinoma", "carcinoma_any")
    for c in cols:
        out[c] = False
    for name, idx in out.groupby("name").groups.items():
        sq = load_squares(annotations / f"{name}.json")
        flags = tile_flags(
            sq, out.loc[idx, "x"].to_numpy(), out.loc[idx, "y"].to_numpy()
        )
        for c in cols:
            out.loc[idx, c] = flags[c]
    return out


def build_features(
    emb_dir: Path,
    slide_table: pd.DataFrame,
    splits: Iterable[str],
    config: LabelConfig,
    cache_root: Path,
    dim: int = 1536,
    batch_size: int = 8192,
    overwrite: bool = False,
    mode: str = "labelled",
    annotations: Path | None = None,
    model: str | None = None,
) -> Path:
    """Build (or reuse) the cache for ``splits``; returns its directory.

    ``mode="labelled"`` (default, for training) keeps label 0/1 tiles only.
    ``mode="all"`` keeps every tile of the splits, including excluded ones
    (label -1), and also stores ``reasons``, ``lg`` and ``hg``; it is meant
    for evaluation and is refused by the default loader.

    ``annotations`` (dir of ``<slide>.json`` squares) is required when the label
    config uses the strict-normal / carcinoma rules (``config.needs_annotations``);
    the flags are then computed exactly from the squares (see huncrc_annotations).

    ``model`` names the embedding (default: the name of ``emb_dir``, e.g.
    ``.../embeddings/prov-gigapath``); an explicit name must equal that dir
    name. ``dim`` is checked against the first embedding before anything is built.
    """
    if config.needs_annotations and annotations is None:
        raise ValueError("this label config needs annotations=<dir of slide JSONs>")
    emb_dir = Path(emb_dir)
    if model is None:
        model = emb_dir.name
    elif model != emb_dir.name:
        raise EmbeddingMismatchError(
            f"REFUSED: embedding model {model!r} but the embeddings dir is {emb_dir} "
            f"(named {emb_dir.name!r}); the dir name must be the model name"
        )
    splits = tuple(sorted(set(splits)))
    wanted = split_slide_ids(slide_table, splits)  # refuses test_final
    key = cache_key(config, splits, wanted["name"], mode, model=model, dim=dim)
    out = Path(cache_root) / key
    if (out / "meta.json").is_file() and not overwrite:
        check_cache_meta(
            json.loads((out / "meta.json").read_text()), config, model, dim
        )
        return out
    _check_embedding_dim(emb_dir / f"{wanted['name'].iloc[0]}.parquet", dim)

    # pass 1: labels from cheap columns, per-slide files of `splits` only
    tiles = load_tiles(emb_dir, slide_table, splits)
    flag_counts: dict[str, int] = {}
    if config.needs_annotations:
        assert annotations is not None
        tiles = attach_annotation_flags(tiles, Path(annotations))
        flag_counts = {
            k: int(tiles[k].sum())
            for k in ("strict_normal", "carcinoma", "carcinoma_any")
        }
        flag_counts["carcinoma_and_lghg"] = int(
            (
                tiles["carcinoma"]
                & (np.maximum(tiles[huncrc_labels.LG], tiles[huncrc_labels.HG]) > 0)
            ).sum()
        )
    tiles = assign_labels(tiles, slide_table, config)
    if mode == "all":
        keep = np.ones(len(tiles), dtype=bool)
    else:
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
    if mode == "all":
        arrays["reasons"] = kept["reason"].fillna("").to_numpy().astype("U20")
        arrays["lg"] = kept[huncrc_labels.LG].to_numpy(np.float32)
        arrays["hg"] = kept[huncrc_labels.HG].to_numpy(np.float32)
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
        "embedding_model": model,
        "n_pos": int((arrays["labels"] == 1).sum()),
        "n_neg": int((arrays["labels"] == 0).sum()),
        "tiles": mode,
        "source": str(emb_dir),
        "annotations": None if annotations is None else str(annotations),
        "annotation_flag_counts": flag_counts,
        "n_by_reason": {
            str(k): int(v)
            for k, v in tiles["reason"].fillna("labelled").value_counts().items()
        },
        "n_neg_by_source": {
            str(k): int(v)
            for k, v in kept["neg_source"].dropna().value_counts().items()
        },
        "split_fold_source": "slides.parquet",
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def load_features(
    cache_dir: Path,
    expected_config: LabelConfig | None = None,
    mmap: bool = True,
    mode: str = "labelled",
    expected_model: str | None = None,
    expected_dim: int | None = None,
) -> FeatureSet:
    """Load a cache, verifying what is expected of it.

    With ``expected_config`` its label rules are verified, with
    ``expected_model`` / ``expected_dim`` its embedding.

    The cache and label-rule versions are always checked against the code.
    ``mode`` must match how the cache was built, so an all-tile evaluation
    cache (labels -1 included) can never be loaded for training by accident.
    """
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    check_cache_meta(meta, expected_config, expected_model, expected_dim)
    built = meta.get("tiles", "labelled")  # caches before the all-tile mode
    if built != mode:
        raise ValueError(f"{cache_dir} is a {built!r}-tile cache, expected {mode!r}")
    features = np.load(cache_dir / "features.npy", mmap_mode="r" if mmap else None)
    names = _ARRAYS + (_ALL_TILE_ARRAYS if mode == "all" else ())
    arrays = {k: np.load(cache_dir / f"{k}.npy") for k in names}
    if mode == "labelled" and not np.isin(arrays["labels"], (0, 1)).all():
        raise ValueError(f"corrupt cache {cache_dir}: labels outside 0/1")
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
