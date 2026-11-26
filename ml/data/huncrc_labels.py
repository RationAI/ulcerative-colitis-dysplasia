"""Tile labels for the HunCRC dysplasia classifier (temporary IBD-45 stand-in).

Pure pandas; no torch, no I/O in the labelling itself. Three pieces:

* :func:`load_slide_table` -- slides.parquet joined 1:1 with the slide-level
  CSV. This is the ONLY source of ``split`` and ``fold``.
* :func:`attach_split` -- puts that split/fold onto a tile frame and refuses a
  tile frame that still carries its own ``split``/``fold`` (the embedding
  parquets have a ``fold`` column that is wrong for 189 of 200 slides).
* :func:`assign_labels` -- the label rules below.

Coverage
    ``lowgrade_dysplasia`` / ``highgrade_dysplasia`` are the fractions of the
    tile's CENTRAL HALF (112x112 of 224x224 at level 2) covered by the union of
    that class's polygons. ``annotation`` is their SUM (can reach 2.0 where LG
    and HG polygons overlap) and is never used here. A tile's dysplasia
    coverage is ``max(LG, HG)``.

Rules (precedence top to bottom)
    1. ``excluded_slide``: slide in ``exclude_slides`` or (``exclude_serrated``)
       a serrated polyp type. Nothing from it is used in training.
    2. coverage >= ``min_coverage`` (and > 0)        -> positive, label 1.
    3. 0 < coverage < ``min_coverage``               -> ``ambiguous_coverage``.
    4. coverage == 0 on a safe-category slide        -> negative, label 0.
    5. coverage == 0 on any other slide              -> ``not_safe_slide``.

Every tile is kept; excluded ones get ``label = -1`` and a ``reason`` so they
remain available for evaluation reporting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from collections.abc import Iterable


LG = "lowgrade_dysplasia"
HG = "highgrade_dysplasia"
SPLIT_COLUMNS = ("split", "fold")

REASON_EXCLUDED_SLIDE = "excluded_slide"
REASON_AMBIGUOUS = "ambiguous_coverage"
REASON_NOT_SAFE = "not_safe_slide"
REASON_CARCINOMA = "carcinoma"
REASONS = (REASON_EXCLUDED_SLIDE, REASON_CARCINOMA, REASON_AMBIGUOUS, REASON_NOT_SAFE)

#: per-tile annotation flags needed by the optional rules (see huncrc_annotations)
STRICT_NORMAL = "strict_normal"
CARCINOMA = "carcinoma"

LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0
LABEL_EXCLUDED = -1

#: Bump whenever the labelling LOGIC changes (precedence, coverage definition,
#: serrated detection, ...). Together with the parameters it forms the rule
#: fingerprint that keys every feature cache, so a changed rule can never
#: silently reuse a cache built under the old one.
LABEL_RULES_VERSION = 1


@dataclass(frozen=True)
class LabelConfig:
    min_coverage: float = 0.0
    exclude_slides: frozenset[str] = field(
        default_factory=lambda: frozenset({"043", "140", "135"})
    )
    safe_categories: frozenset[str] = field(
        default_factory=lambda: frozenset({"negative", "non_neoplastic_lesion"})
    )
    exclude_serrated: bool = True
    #: Step 8: "strict normal" tiles (central half fully inside normal squares, no
    #: LG/HG anywhere in the tile) of POSITIVE slides become negatives.
    within_slide_negatives: bool = False
    #: Step 8: carcinoma tiles (central half fully inside adenocarcinoma /
    #: suspicious_for_invasion / tumor_necrosis squares) never enter training.
    exclude_carcinoma: bool = False

    @property
    def needs_annotations(self) -> bool:
        return self.within_slide_negatives or self.exclude_carcinoma

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be in [0, 1], got {self.min_coverage}")
        object.__setattr__(
            self, "exclude_slides", frozenset(map(str, self.exclude_slides))
        )
        object.__setattr__(self, "safe_categories", frozenset(self.safe_categories))
        for s in self.exclude_slides:
            _check_slide_name(s)


def rules_fingerprint(config: LabelConfig) -> str:
    """Short hash of the label-rule parameters and ``LABEL_RULES_VERSION``."""
    payload = {
        "label_rules_version": LABEL_RULES_VERSION,
        "min_coverage": float(config.min_coverage),
        "exclude_slides": sorted(config.exclude_slides),
        "safe_categories": sorted(config.safe_categories),
        "exclude_serrated": bool(config.exclude_serrated),
    }
    #: optional rules enter the fingerprint only when switched on, so caches built
    #: under the default rules keep their keys
    for opt in ("within_slide_negatives", "exclude_carcinoma"):
        if getattr(config, opt):
            payload[opt] = True
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def rules_diff(
    config: LabelConfig, reference: LabelConfig
) -> dict[str, tuple[object, object]]:
    """``{field: (config value, reference value)}`` for every field that differs."""
    out: dict[str, tuple[object, object]] = {}
    for f in fields(LabelConfig):
        a, b = getattr(config, f.name), getattr(reference, f.name)
        if a != b:
            out[f.name] = (
                sorted(a) if isinstance(a, frozenset) else a,
                sorted(b) if isinstance(b, frozenset) else b,
            )
    return out


def _check_slide_name(name: str) -> None:
    if not (isinstance(name, str) and len(name) == 3 and name.isdigit()):
        raise ValueError(f"slide names are zero-padded 3-digit strings, got {name!r}")


def load_slide_table(slides: pd.DataFrame, slide_csv: pd.DataFrame) -> pd.DataFrame:
    """Join slides.parquet (``id, path, split, fold``) with the slide-level CSV.

    ``slide_csv`` must be read with ``dtype=str`` so ``slideID`` stays "001";
    an int column is rejected rather than silently zero-padded.
    Returns ``name, id, split, fold, category, polyp_type``.
    """
    if (
        not pd.api.types.is_string_dtype(slide_csv["slideID"])
        or slide_csv["slideID"].map(lambda v: not isinstance(v, str)).any()
    ):
        raise TypeError("slideID must be read as str (pd.read_csv(..., dtype=str))")

    s = slides[["id", "path", "split", "fold"]].copy()
    s["name"] = s["path"].map(lambda p: Path(str(p)).name.split(".")[0])
    c = slide_csv.rename(
        columns={"slideID": "name", "CATEGORY": "category", "Polyp type": "polyp_type"}
    )[["name", "category", "polyp_type"]]

    for names in (s["name"], c["name"]):
        for n in names:
            _check_slide_name(n)
        if names.duplicated().any():
            raise ValueError(
                f"duplicate slide names: {names[names.duplicated()].tolist()}"
            )
    if set(s["name"]) != set(c["name"]):
        raise ValueError(
            "slides.parquet and the CSV do not cover the same slides: "
            f"only parquet={sorted(set(s['name']) - set(c['name']))}, "
            f"only csv={sorted(set(c['name']) - set(s['name']))}"
        )

    out = s.merge(c, on="name", how="inner", validate="1:1")
    assert len(out) == len(s) == len(c)
    return out[["name", "id", "split", "fold", "category", "polyp_type"]]


def attach_split(tiles: pd.DataFrame, slide_table: pd.DataFrame) -> pd.DataFrame:
    """Add ``name, split, fold`` from the slide table to a tile frame.

    The tile frame must NOT already have ``split``/``fold``: those come from
    slides.parquet only, never from the tiling or embedding parquets.
    """
    leaked = [c for c in SPLIT_COLUMNS if c in tiles.columns]
    assert not leaked, (
        f"tile frame carries {leaked}; split/fold must come from slides.parquet "
        "only -- do not read these columns from the tile/embedding parquets"
    )
    out = tiles.merge(
        slide_table[["id", "name", "split", "fold"]].rename(columns={"id": "slide_id"}),
        on="slide_id",
        how="left",
        validate="many_to_one",
    )
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "slide_id"].unique()[:5]
        raise ValueError(f"tiles from slides not in the slide table: {list(missing)}")
    return out


FORBIDDEN_SPLITS = frozenset({"test_final"})


def split_slide_ids(
    slide_table: pd.DataFrame, splits: Iterable[str], allow_test_final: bool = False
) -> pd.DataFrame:
    """``id, name`` of the slides in ``splits`` (from slides.parquet only).

    test_final is refused unless ``allow_test_final`` is set explicitly.
    """
    splits = set(splits)
    unknown = splits - set(slide_table["split"])
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}")
    if splits & FORBIDDEN_SPLITS and not allow_test_final:
        raise PermissionError("test_final may not be read (allow_test_final=False)")
    return slide_table.loc[
        slide_table["split"].isin(splits), ["id", "name"]
    ].sort_values("name")


def read_split_rows(
    path: Path, slide_ids: Iterable[str], columns: list[str]
) -> pd.DataFrame:
    """Read ``columns`` of one parquet, keeping only rows of ``slide_ids``.

    The filter is applied by pyarrow at read time, so rows of other slides (e.g.
    test_final rows in a multi-slide tiling shard) are never materialised.
    """
    ids = list(slide_ids)
    cols = list(dict.fromkeys(["slide_id", *columns]))
    df = pd.read_parquet(path, columns=cols, filters=[("slide_id", "in", ids)])
    assert df["slide_id"].isin(ids).all()
    return df[list(dict.fromkeys(columns))]


def load_tiles(
    tile_dir: Path | None,
    slide_table: pd.DataFrame,
    splits: Iterable[str],
    columns: Iterable[str] = ("slide_id", "x", "y", LG, HG),
    files: Iterable[Path] | None = None,
) -> pd.DataFrame:
    """Load tiles of ``splits`` only, with split/fold attached from the slide table.

    Two layouts:

    * per-slide files ``<tile_dir>/<name>.parquet`` (embeddings): only the files
      of the requested slides are opened;
    * ``files=`` (e.g. the 18 multi-slide tiling shards): every file is opened but
      rows are filtered on slide_id at read time, before any statistic or label.

    ``split``/``fold`` may not be requested from the files.
    """
    columns = list(columns)
    assert not set(columns) & set(SPLIT_COLUMNS), (
        "never read split/fold from tile/embedding parquets; they come from slides.parquet"
    )
    if "slide_id" not in columns:
        columns = ["slide_id", *columns]
    wanted = split_slide_ids(slide_table, splits)
    ids = wanted["id"].tolist()
    if files is not None:
        paths = sorted(Path(f) for f in files)
    else:
        assert tile_dir is not None, "give tile_dir or files"
        paths = [Path(tile_dir) / f"{n}.parquet" for n in wanted["name"]]
    frames = [read_split_rows(p, ids, columns) for p in paths]
    tiles = attach_split(pd.concat(frames, ignore_index=True), slide_table)
    assert set(tiles["split"]) <= set(splits)
    return tiles


def excluded_slide_names(slide_table: pd.DataFrame, config: LabelConfig) -> set[str]:
    excluded = set(config.exclude_slides)
    if config.exclude_serrated:
        serrated = (
            slide_table["polyp_type"]
            .astype(str)
            .str.contains("serrated", case=False, na=False)
        )
        excluded |= set(slide_table.loc[serrated, "name"])
    return excluded


def safe_slide_names(slide_table: pd.DataFrame, config: LabelConfig) -> set[str]:
    """Slides whose zero-coverage tiles may serve as negatives."""
    safe = slide_table.loc[slide_table["category"].isin(config.safe_categories), "name"]
    return set(safe) - excluded_slide_names(slide_table, config)


def assign_labels(
    tiles: pd.DataFrame, slide_table: pd.DataFrame, config: LabelConfig | None = None
) -> pd.DataFrame:
    """Return ``tiles`` plus ``coverage``, ``label`` (1/0/-1), ``reason``, ``neg_source``.

    ``tiles`` needs ``name`` (see :func:`attach_split`), ``LG`` and ``HG``; with
    ``within_slide_negatives`` also ``strict_normal``, with ``exclude_carcinoma``
    also ``carcinoma`` (bool columns). ``reason`` is None for training tiles;
    ``neg_source`` is "safe_slide" / "positive_slide_normal" for negatives.

    Precedence: excluded slide > carcinoma (if excluded) > positive > ambiguous >
    negative on a safe slide > strict-normal negative on a positive slide (if
    enabled) > not_safe_slide. A positive slide is a non-excluded slide with any
    tile of LG/HG coverage > 0 in ``tiles``, so pass whole slides.
    """
    config = config or LabelConfig()
    required = ["name", LG, HG]
    if config.within_slide_negatives:
        required.append(STRICT_NORMAL)
    if config.exclude_carcinoma:
        required.append(CARCINOMA)
    for col in required:
        if col not in tiles.columns:
            raise KeyError(f"tiles is missing column {col!r}")
    lg = tiles[LG].to_numpy(dtype=float)
    hg = tiles[HG].to_numpy(dtype=float)
    if np.isnan(lg).any() or np.isnan(hg).any():
        raise ValueError("NaN in LG/HG coverage")
    if (
        (lg < 0).any()
        or (hg < 0).any()
        or (lg > 1 + 1e-9).any()
        or (hg > 1 + 1e-9).any()
    ):
        raise ValueError("LG/HG coverage outside [0, 1]")

    coverage = np.maximum(lg, hg)
    is_excluded = (
        tiles["name"].isin(excluded_slide_names(slide_table, config)).to_numpy()
    )
    is_safe = tiles["name"].isin(safe_slide_names(slide_table, config)).to_numpy()

    has_dysplasia = coverage > 0
    positive = has_dysplasia & (coverage >= config.min_coverage)
    ambiguous = has_dysplasia & ~positive

    label = np.full(len(tiles), LABEL_EXCLUDED, dtype=np.int8)
    reason = np.full(len(tiles), None, dtype=object)

    source = np.full(len(tiles), None, dtype=object)

    # Assigned in reverse precedence so the strongest rule wins.
    reason[~has_dysplasia & ~is_safe] = REASON_NOT_SAFE
    if config.within_slide_negatives:
        strict = tiles[STRICT_NORMAL].to_numpy(dtype=bool)
        if (strict & has_dysplasia).any():
            raise ValueError("a strict-normal tile has LG/HG coverage")
        pos_slide = (
            pd.Series(has_dysplasia, index=tiles.index)
            .groupby(tiles["name"])
            .transform("any")
        ).to_numpy(dtype=bool) & ~is_excluded
        wsn = strict & pos_slide & ~is_safe & ~has_dysplasia
        label[wsn] = LABEL_NEGATIVE
        reason[wsn] = None
        source[wsn] = "positive_slide_normal"
    neg = ~has_dysplasia & is_safe
    label[neg] = LABEL_NEGATIVE
    reason[neg] = None
    source[neg] = "safe_slide"
    reason[ambiguous] = REASON_AMBIGUOUS
    label[positive] = LABEL_POSITIVE
    reason[positive] = None
    if config.exclude_carcinoma:
        carc = tiles[CARCINOMA].to_numpy(dtype=bool)
        label[carc] = LABEL_EXCLUDED
        reason[carc] = REASON_CARCINOMA
    label[is_excluded] = LABEL_EXCLUDED
    reason[is_excluded] = REASON_EXCLUDED_SLIDE
    source[label != LABEL_NEGATIVE] = None

    out = tiles.copy()
    out["coverage"] = coverage
    out["label"] = label
    out["reason"] = pd.Series(reason, index=tiles.index, dtype=object)
    out["neg_source"] = pd.Series(source, index=tiles.index, dtype=object)
    assert ((out["label"] == LABEL_EXCLUDED) == out["reason"].notna()).all()
    return out


def summarize(
    labelled: pd.DataFrame, by: Iterable[str] = ("split", "fold")
) -> pd.DataFrame:
    """Per-group positives, negatives, ratio, contributing slides, excluded by reason."""
    by = list(by)
    g = labelled.assign(
        pos=labelled["label"] == LABEL_POSITIVE,
        neg=labelled["label"] == LABEL_NEGATIVE,
        **{r: labelled["reason"] == r for r in REASONS},
    ).groupby(by, dropna=False)
    out = g[["pos", "neg", *REASONS]].sum()
    out["neg_per_pos"] = out["neg"] / out["pos"].where(out["pos"] > 0)
    out["pos_slides"] = g.apply(
        lambda d: d.loc[d["pos"], "name"].nunique(), include_groups=False
    )
    out["neg_slides"] = g.apply(
        lambda d: d.loc[d["neg"], "name"].nunique(), include_groups=False
    )
    return out[["pos", "neg", "neg_per_pos", "pos_slides", "neg_slides", *REASONS]]
