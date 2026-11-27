"""Training and validation rows of one CV fold of a HunCRC feature cache.

Negatives are capped per slide in training only; validation keeps every tile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ml.data.huncrc_features import FeatureSet


def subsample_per_slide(
    rows: np.ndarray, names: np.ndarray, cap: int | None, rng: np.random.Generator
) -> np.ndarray:
    """At most ``cap`` rows per slide (random, sorted output); None = keep all."""
    if cap is None:
        return np.sort(rows)
    keep = []
    for n in np.unique(names[rows]):
        r = rows[names[rows] == n]
        keep.append(r if len(r) <= cap else rng.choice(r, cap, replace=False))
    return np.sort(np.concatenate(keep)) if keep else rows[:0]


def neg_source(fs: FeatureSet, safe: set[str]) -> np.ndarray:
    """'safe_slide' / 'positive_slide_normal' for label-0 rows, '' otherwise."""
    is_safe = np.isin(fs.slide_names, list(safe))
    out = np.full(len(fs), "", dtype=object)
    out[(fs.labels == 0) & is_safe] = "safe_slide"
    out[(fs.labels == 0) & ~is_safe] = "positive_slide_normal"
    return out


def fold_rows(
    fs: FeatureSet,
    k: int,
    smoke: int | None,
    neg_cap: int | None,
    cap_seed: int,
    safe: set[str],
):
    tr = np.flatnonzero((fs.folds >= 0) & (fs.folds != k))
    val = np.flatnonzero(fs.folds == k)
    assert not set(fs.slide_names[tr]) & set(fs.slide_names[val]), "slide overlap"
    rng = np.random.default_rng([cap_seed, k])
    if smoke:
        tr = subsample_per_slide(tr, fs.slide_names, smoke, rng)
        val = subsample_per_slide(val, fs.slide_names, smoke, rng)
    pos = tr[fs.labels[tr] == 1]
    neg = subsample_per_slide(tr[fs.labels[tr] == 0], fs.slide_names, neg_cap, rng)
    train = np.sort(np.concatenate([pos, neg]))
    src = neg_source(fs, safe)

    def counts(r: np.ndarray) -> dict[str, int]:
        return {
            "pos": int((fs.labels[r] == 1).sum()),
            "neg_safe_slide": int((src[r] == "safe_slide").sum()),
            "neg_positive_slide_normal": int((src[r] == "positive_slide_normal").sum()),
            "slides": len(set(fs.slide_names[r])),
            "slides_with_pos_slide_normal": len(
                set(fs.slide_names[r][src[r] == "positive_slide_normal"])
            ),
        }

    return (
        train,
        val,
        {"train_before_cap": counts(tr), "train": counts(train), "val": counts(val)},
    )
