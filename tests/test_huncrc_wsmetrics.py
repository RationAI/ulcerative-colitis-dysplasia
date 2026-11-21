"""Known-answer tests for ml.huncrc_wsmetrics on the synthetic cohort."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.huncrc_features import FeatureSet, load_features
from ml.data.huncrc_labels import LabelConfig, load_slide_table
from ml.huncrc_wsmetrics import PrelimMasks, prelim_masks, prelim_report
from tests.huncrc_synth import build


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[FeatureSet, PrelimMasks]:
    tmp = tmp_path_factory.mktemp("ws")
    c = build(tmp)
    st = load_slide_table(
        pd.read_parquet(c["tiling"] / "slides" / "slides.parquet"),
        pd.read_csv(c["slide_csv"], dtype=str),
    )
    fs = load_features(
        c["test_preliminary_all"], expected_config=LabelConfig(), mode="all"
    )
    return fs, prelim_masks(fs, st, Path(c["annotations"]))


def test_masks(setup: tuple[FeatureSet, PrelimMasks]) -> None:
    _, m = setup
    # prelim: 2 adenoma (pos), negative + hyperplastic (safe), CRC without LG (other)
    assert sorted(m.cls.tolist()) == ["other", "pos", "pos", "safe", "safe"]
    pos = m.cls[m.slide_of] == "pos"
    assert (m.strict & pos).sum() == 2 * 28 and (m.dys & pos).sum() == 2 * 30
    assert m.carcinoma.sum() > 0 and not (m.carcinoma & m.normal).any()


def test_perfect_scores(setup: tuple[FeatureSet, PrelimMasks]) -> None:
    _, m = setup
    s = np.where(m.dys, 0.9, np.where(m.normal, 0.1, 0.5))
    rep = prelim_report(
        np.stack([s, s]), np.array([0.5, 0.5]), m, n_boot=20, min_per_class=10
    )
    r = {k: v["mean"] for k, v in rep["summary"].items()}
    assert r["ws_auroc"] == 1.0 and r["spec_normal_pos"] == 1.0
    assert r["frac_above_normal_pos"] == 0.0 and r["frac_above_normal_safe"] == 0.0
    assert r["old_auroc"] == 1.0 and r["per_slide_ws_auroc_median"] == 1.0
    assert r["frac_above_carcinoma"] == 1.0 and r["frac_above_normal_crc"] == 0.0
    assert r["slide_auroc_mean_all"] == 1.0
    assert rep["summary"]["ws_auroc"]["sd"] == 0.0


def test_slide_type_shortcut_is_exposed(setup: tuple[FeatureSet, PrelimMasks]) -> None:
    _, m = setup
    # a "lesion slide" detector: every tile of a positive slide scores high
    s = np.where(m.cls[m.slide_of] == "pos", 0.9, 0.1)
    rep = prelim_report(s[None], np.array([0.5]), m, n_boot=0)
    r = {k: v["mean"] for k, v in rep["summary"].items()}
    assert r["ws_auroc"] == 0.5  # no within-slide discrimination at all
    assert r["frac_above_normal_pos"] == 1.0 and r["frac_above_normal_safe"] == 0.0
    assert r["old_auroc"] == 1.0  # ...yet the old-style tile AUROC looks perfect
    assert r["spec_normal_pos"] == 0.0
