"""Step-7 metrics on test_preliminary for any set of fold models.

Given per-fold scores of EVERY test_preliminary tile (all-tile cache order) and
each fold's validation threshold, computes per fold:

(i)   within-slide: dysplasia vs strict-normal tiles of the positive slides
      (pooled AUROC / AUPRC, median per-slide AUROC over slides with >= 50 of
      each), specificity on normal and on strict-normal tiles of positive slides;
(ii)  shortcut test: median score and fraction above threshold of normal tiles
      in positive slides, normal tiles in safe slides, label-0 tiles of safe slides;
(iii) old-style tile AUROC / AUPRC: dysplasia vs safe-slide label-0 tiles;
(iv)  slide AUROC, positive (contains LG/HG) vs safe slides, for three
      aggregations: mean of all tiles, mean of the top 1 %, fraction above threshold;
(v)   carcinoma: fraction above threshold on carcinoma tiles vs normal tiles of
      the CRC slides that have carcinoma squares.

Reported as mean +/- sd over the folds, plus a 95 % CI of the fold mean from a
bootstrap over slides (stratified by slide class: pos / safe / other).
Definitions follow Step 7 (coverage from the annotation squares, see
``ml.data.huncrc_annotations.tile_flags``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ml.data.huncrc_annotations import load_squares, tile_flags
from ml.data.huncrc_labels import LabelConfig, safe_slide_names
from ml.huncrc_eval import SortedScores, weighted_auroc_ap


if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from ml.data.huncrc_features import FeatureSet


MIN_PER_CLASS = 50
TOP_FRACTION = 0.01


@dataclass
class PrelimMasks:
    names: np.ndarray  # (S,) slide names
    slide_of: np.ndarray  # (N,) slide index per tile
    cls: np.ndarray  # (S,) "pos" | "safe" | "other"
    category: np.ndarray  # (S,)
    dys: np.ndarray  # (N,) bool, label 1
    label0_safe: np.ndarray  # (N,) bool, label 0 on a safe slide
    normal: np.ndarray
    strict: np.ndarray
    carcinoma: np.ndarray


def prelim_masks(
    fs: FeatureSet,
    slide_table: pd.DataFrame,
    annotations: Path,
    config: LabelConfig | None = None,
) -> PrelimMasks:
    """Tile masks for an ALL-tile test_preliminary cache (default label rules)."""
    config = config or LabelConfig()
    names, slide_of = np.unique(fs.slide_names, return_inverse=True)
    n = len(fs)
    flags = {k: np.zeros(n, bool) for k in ("normal", "strict_normal", "carcinoma")}
    for j, name in enumerate(names):
        m = slide_of == j
        f = tile_flags(load_squares(annotations / f"{name}.json"), fs.x[m], fs.y[m])
        for k in flags:
            flags[k][m] = f[k]
    dys = fs.labels == 1
    if (flags["normal"] & dys).any():
        raise ValueError("a tile is both normal and dysplasia")
    safe = safe_slide_names(slide_table, config)
    has_lghg = np.bincount(slide_of, weights=dys, minlength=len(names)) > 0
    cls = np.where(
        has_lghg, "pos", np.where(np.isin(names, list(safe)), "safe", "other")
    )
    stn = slide_table.set_index("name")
    return PrelimMasks(
        names=names,
        slide_of=slide_of,
        cls=cls,
        category=stn.loc[names, "category"].to_numpy(),
        dys=dys,
        label0_safe=(fs.labels == 0) & (cls[slide_of] == "safe"),
        normal=flags["normal"],
        strict=flags["strict_normal"],
        carcinoma=flags["carcinoma"],
    )


class _FoldStats:
    """Everything one fold model needs, with weighted (slide-bootstrap) versions."""

    def __init__(
        self,
        score: np.ndarray,
        thr: float,
        m: PrelimMasks,
        min_per_class: int = MIN_PER_CLASS,
    ) -> None:
        self.m = m
        self.min_per_class = min_per_class
        self.score = score
        self.pred = score >= thr
        pos_slide = m.cls[m.slide_of] == "pos"
        safe_slide = m.cls[m.slide_of] == "safe"
        # (i) within-slide, positive slides that have strict-normal tiles
        with_strict = (
            np.bincount(
                m.slide_of, weights=m.strict & pos_slide, minlength=len(m.names)
            )
            > 0
        )
        self.ws_items = np.flatnonzero(with_strict[m.slide_of] & (m.dys | m.strict))
        self.ws = SortedScores(m.dys[self.ws_items].astype(int), score[self.ws_items])
        self.normal_pos = np.flatnonzero(m.normal & pos_slide)
        self.strict_pos = np.flatnonzero(m.strict & pos_slide)
        self.normal_safe = np.flatnonzero(m.normal & safe_slide)
        self.label0_safe = np.flatnonzero(m.label0_safe)
        # (iii) old-style
        self.old_items = np.flatnonzero(m.dys | m.label0_safe)
        self.old = SortedScores(
            m.dys[self.old_items].astype(int), score[self.old_items]
        )
        # (iv) slide aggregations over ALL tiles, pos vs safe
        n_per = np.bincount(m.slide_of)
        self.agg = {
            "mean_all": np.bincount(m.slide_of, weights=score) / n_per,
            "frac_above_thr": np.bincount(m.slide_of, weights=self.pred) / n_per,
            "top1pct_mean": np.array(
                [
                    np.sort(score[m.slide_of == j])[
                        -max(1, math.ceil(TOP_FRACTION * n_per[j])) :
                    ].mean()
                    for j in range(len(m.names))
                ]
            ),
        }
        self.slide_main = np.flatnonzero(m.cls != "other")
        y_slide = (m.cls[self.slide_main] == "pos").astype(int)
        self.slide_sorted = {
            k: SortedScores(y_slide, v[self.slide_main]) for k, v in self.agg.items()
        }
        # (v) carcinoma vs normal in CRC slides with carcinoma squares
        crc = (m.category == "CRC") & (
            np.bincount(m.slide_of, weights=m.carcinoma, minlength=len(m.names)) > 0
        )
        self.carc_items = np.flatnonzero(m.carcinoma & crc[m.slide_of])
        self.carc_normal_items = np.flatnonzero(m.normal & crc[m.slide_of])
        self.crc_slides = m.names[crc]

    def _rate(self, items: np.ndarray, w_tile: np.ndarray, above: bool = True) -> float:
        if len(items) == 0:
            return float("nan")
        w = w_tile[items]
        hit = self.pred[items] if above else ~self.pred[items]
        return float((w * hit).sum() / w.sum()) if w.sum() > 0 else float("nan")

    def metrics(self, w_slide: np.ndarray | None = None) -> dict[str, float]:
        m = self.m
        if w_slide is None:
            w_slide = np.ones(len(m.names))
        w_tile = w_slide[m.slide_of]
        ws_auroc, ws_auprc = self.ws.metrics(w_tile[self.ws_items][self.ws.order])
        old_auroc, old_auprc = self.old.metrics(w_tile[self.old_items][self.old.order])
        out = {
            "ws_auroc": ws_auroc,
            "ws_auprc": ws_auprc,
            "spec_normal_pos": self._rate(self.normal_pos, w_tile, above=False),
            "spec_strict_pos": self._rate(self.strict_pos, w_tile, above=False),
            "frac_above_normal_pos": self._rate(self.normal_pos, w_tile),
            "frac_above_normal_safe": self._rate(self.normal_safe, w_tile),
            "frac_above_label0_safe": self._rate(self.label0_safe, w_tile),
            "old_auroc": old_auroc,
            "old_auprc": old_auprc,
            "frac_above_carcinoma": self._rate(self.carc_items, w_tile),
            "frac_above_normal_crc": self._rate(self.carc_normal_items, w_tile),
        }
        ws_main = w_slide[self.slide_main]
        for k, srt in self.slide_sorted.items():
            out[f"slide_auroc_{k}"] = srt.metrics(ws_main[srt.order])[0]
        return out

    def unweighted_extras(self) -> dict[str, Any]:
        """Point-only statistics (medians, per-slide AUROCs)."""
        m, s = self.m, self.score
        per_slide = {}
        for j in np.flatnonzero(m.cls == "pos"):
            mj = m.slide_of == j
            nd, nn = int((mj & m.dys).sum()), int((mj & m.strict).sum())
            if nd >= self.min_per_class and nn >= self.min_per_class:
                it = np.flatnonzero(mj & (m.dys | m.strict))
                per_slide[str(m.names[j])] = weighted_auroc_ap(
                    m.dys[it].astype(int), s[it]
                )[0]

        def med(items: np.ndarray) -> float:
            return float(np.median(s[items])) if len(items) else float("nan")

        return {
            "per_slide_ws_auroc": per_slide,
            "per_slide_ws_auroc_median": float(np.median(list(per_slide.values())))
            if per_slide
            else float("nan"),
            "median_normal_pos": med(self.normal_pos),
            "median_normal_safe": med(self.normal_safe),
            "median_label0_safe": med(self.label0_safe),
        }


CI_METRICS = (
    "ws_auroc",
    "ws_auprc",
    "spec_normal_pos",
    "frac_above_normal_pos",
    "frac_above_normal_safe",
    "old_auroc",
    "slide_auroc_mean_all",
    "slide_auroc_top1pct_mean",
    "slide_auroc_frac_above_thr",
    "frac_above_carcinoma",
    "frac_above_normal_crc",
)


def prelim_report(
    scores: np.ndarray,
    thresholds: np.ndarray,
    m: PrelimMasks,
    n_boot: int = 1000,
    seed: int = 0,
    min_per_class: int = MIN_PER_CLASS,
) -> dict[str, Any]:
    """Per-fold metrics, mean +/- sd over folds, and slide-bootstrap CIs of the mean.

    ``scores`` is (folds, N) in all-tile cache order; ``thresholds`` is (folds,).
    """
    scores = np.asarray(scores)
    folds = [
        _FoldStats(s, float(t), m, min_per_class)
        for s, t in zip(scores, thresholds, strict=True)
    ]
    per_fold = [
        {
            **f.metrics(),
            **{
                k: v
                for k, v in f.unweighted_extras().items()
                if k != "per_slide_ws_auroc"
            },
        }
        for f in folds
    ]
    keys = list(per_fold[0])
    summary: dict[str, dict[str, Any]] = {}
    for k in keys:
        v = np.array([p[k] for p in per_fold], dtype=float)
        summary[k] = {
            "mean": float(np.nanmean(v)),
            "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"),
        }
    strata = np.select([m.cls == "pos", m.cls == "safe"], [0, 1], 2)
    # one bootstrap loop for all metrics: slides resampled within their class,
    # every fold model evaluated on the same resample, then averaged over folds
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    draws: dict[str, list[float]] = {k: [] for k in CI_METRICS}
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # empty groups in a resample -> NaN
        for _ in range(n_boot):
            c = np.zeros(len(m.names))
            for g in groups:
                np.add.at(c, rng.choice(g, len(g)), 1)
            per = [f.metrics(c) for f in folds]
            for k in CI_METRICS:
                draws[k].append(float(np.nanmean([p[k] for p in per])))
        for k in CI_METRICS:
            v = np.asarray(draws[k])
            summary[k]["ci"] = (
                [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                if n_boot and np.isfinite(v).any()
                else [float("nan"), float("nan")]
            )
    return {
        "per_fold": per_fold,
        "summary": summary,
        "per_slide_ws_auroc": [
            f.unweighted_extras()["per_slide_ws_auroc"] for f in folds
        ],
        "n": {
            "positive_slides": int((m.cls == "pos").sum()),
            "safe_slides": int((m.cls == "safe").sum()),
            "other_slides": int((m.cls == "other").sum()),
            "dysplasia_tiles": int(m.dys.sum()),
            "strict_normal_pos": len(folds[0].strict_pos),
            "normal_pos": len(folds[0].normal_pos),
            "normal_safe": len(folds[0].normal_safe),
            "carcinoma_crc": len(folds[0].carc_items),
            "crc_slides_with_carcinoma": [str(x) for x in folds[0].crc_slides],
        },
    }
