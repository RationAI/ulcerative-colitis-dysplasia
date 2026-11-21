"""Unit tests for ml.data.huncrc_labels -- synthetic frames only, no real data."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml.data.huncrc_labels import (
    HG,
    LG,
    LabelConfig,
    assign_labels,
    attach_split,
    load_slide_table,
    load_tiles,
    safe_slide_names,
    summarize,
)


def _slides() -> pd.DataFrame:
    rows = [
        # name, split, fold
        ("001", "train", 0.0),
        ("002", "train", 0.0),
        ("003", "train", 1.0),
        ("043", "train", 1.0),
        ("135", "train", 1.0),
        ("140", "train", 0.0),
        ("010", "test_preliminary", None),
        ("020", "test_final", None),
    ]
    return pd.DataFrame(
        {
            "id": [f"h{n}" for n, *_ in rows],
            "path": [f"/data/HunCRC/{n}.mrxs" for n, *_ in rows],
            "split": [s for _, s, _ in rows],
            "fold": [f for *_, f in rows],
        }
    )


def _csv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slideID": ["001", "002", "003", "043", "135", "140", "010", "020"],
            "CATEGORY": [
                "adenoma",
                "negative",
                "CRC",
                "non_neoplastic_lesion",
                "adenoma",
                "non_neoplastic_lesion",
                "non_neoplastic_lesion",
                "negative",
            ],
            "Polyp type": [
                "tubular_adenoma",
                "Not_applicable",
                "Not_applicable",
                "Not_applicable",
                "sessile_serrated",
                "tubular_adenoma",
                "hyperplastic",
                "Not_applicable",
            ],
        },
        dtype=str,
    )


@pytest.fixture
def table() -> pd.DataFrame:
    return load_slide_table(_slides(), _csv())


def _tiles(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slide_id": [f"h{n}" for n, _, _ in rows],
            "x": range(len(rows)),
            "y": 0,
            LG: [lg for _, lg, _ in rows],
            HG: [hg for *_, hg in rows],
        }
    )


# --- slide table ---------------------------------------------------------


def test_slide_table_joins_one_to_one(table: pd.DataFrame) -> None:
    assert len(table) == 8
    assert table.set_index("name").loc["002", "category"] == "negative"
    assert table.set_index("name").loc["003", "split"] == "train"


def test_slide_table_rejects_int_ids() -> None:
    csv = _csv()
    csv["slideID"] = csv["slideID"].astype(int)
    with pytest.raises(TypeError):
        load_slide_table(_slides(), csv)


def test_slide_table_rejects_missing_slide() -> None:
    with pytest.raises(ValueError, match="same slides"):
        load_slide_table(_slides(), _csv().iloc[:-1])


def test_slide_table_rejects_duplicate() -> None:
    csv = pd.concat([_csv(), _csv().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_slide_table(_slides(), csv)


def test_slide_table_rejects_unpadded_names() -> None:
    csv = _csv()
    csv.loc[0, "slideID"] = "1"
    with pytest.raises(ValueError, match="zero-padded"):
        load_slide_table(_slides(), csv)


# --- split / fold provenance ---------------------------------------------


def test_attach_split_uses_slide_table(table: pd.DataFrame) -> None:
    out = attach_split(_tiles([("003", 0, 0), ("010", 0, 0)]), table)
    assert out["fold"].tolist()[0] == 1.0
    assert out["split"].tolist() == ["train", "test_preliminary"]


@pytest.mark.parametrize("col", ["fold", "split"])
def test_attach_split_refuses_tile_side_split(table: pd.DataFrame, col: str) -> None:
    tiles = _tiles([("003", 0, 0)]).assign(**{col: 4.0})
    with pytest.raises(AssertionError, match=r"slides\.parquet"):
        attach_split(tiles, table)


def test_attach_split_unknown_slide(table: pd.DataFrame) -> None:
    tiles = _tiles([("003", 0, 0)]).assign(slide_id="nope")
    with pytest.raises(ValueError, match="not in the slide table"):
        attach_split(tiles, table)


def test_load_tiles_never_opens_other_splits(
    table: pd.DataFrame, tmp_path: Path
) -> None:
    for name in table["name"]:
        if name == "020":  # test_final: deliberately absent; opening it would fail
            continue
        _tiles([(name, 0.0, 0.0)]).assign(fold=9.0).to_parquet(
            tmp_path / f"{name}.parquet"
        )
    out = load_tiles(tmp_path, table, splits=("train", "test_preliminary"))
    assert set(out["split"]) == {"train", "test_preliminary"}
    assert "020" not in set(out["name"])
    # the bogus tile-side fold=9 was not read; fold comes from the slide table
    assert set(out["fold"].dropna()) == {0.0, 1.0}


def test_load_tiles_refuses_split_columns(table: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        load_tiles(tmp_path, table, splits=("train",), columns=("slide_id", "fold"))


# --- label rules ---------------------------------------------------------


def _label(
    table: pd.DataFrame, rows: list[tuple[str, float, float]], **cfg: Any
) -> pd.DataFrame:
    return assign_labels(attach_split(_tiles(rows), table), table, LabelConfig(**cfg))


def test_default_exclusions_and_serrated(table: pd.DataFrame) -> None:
    # 043 & 140 are in the safe categories but excluded -> not safe
    assert safe_slide_names(table, LabelConfig()) == {"002", "010", "020"}
    # serrated 135 excluded even when not listed explicitly
    cfg = LabelConfig(exclude_slides=frozenset())
    out = _label(table, [("135", 0.5, 0)], exclude_slides=frozenset())
    assert out["reason"].tolist() == ["excluded_slide"]
    assert "043" in safe_slide_names(table, cfg)


def test_rules_default(table: pd.DataFrame) -> None:
    out = _label(
        table,
        [
            ("001", 0.3, 0.0),  # LG on adenoma slide -> pos
            ("001", 0.0, 0.0),  # unannotated on adenoma slide -> not safe
            ("003", 0.0, 0.0),  # unannotated on CRC slide -> not safe
            ("003", 0.0, 0.01),  # HG on CRC slide -> pos
            ("002", 0.0, 0.0),  # negative slide -> neg
            ("002", 0.2, 0.0),  # LG on a safe slide -> pos (coverage wins)
            ("010", 0.0, 0.0),  # hyperplastic, test_prelim -> neg
            ("043", 0.9, 0.0),  # excluded slide, even with LG
            ("043", 0.0, 0.0),  # excluded slide
            ("140", 0.0, 0.0),  # excluded slide
            ("135", 0.0, 0.0),  # excluded (serrated)
        ],
    )
    assert out["label"].tolist() == [1, -1, -1, 1, 0, 1, 0, -1, -1, -1, -1]
    assert out["reason"].tolist() == [
        None,
        "not_safe_slide",
        "not_safe_slide",
        None,
        None,
        None,
        None,
        "excluded_slide",
        "excluded_slide",
        "excluded_slide",
        "excluded_slide",
    ]


def test_tiny_coverage_is_positive_by_default(table: pd.DataFrame) -> None:
    out = _label(table, [("001", 1e-6, 0.0)])
    assert out["label"].tolist() == [1]


def test_min_coverage_ambiguous(table: pd.DataFrame) -> None:
    out = _label(
        table,
        [
            ("001", 0.49, 0.0),  # ambiguous
            ("001", 0.5, 0.0),  # boundary: >= -> pos
            ("001", 0.3, 0.6),  # max(LG,HG)=0.6 -> pos
            ("001", 0.3, 0.3),  # max=0.3 -> ambiguous (no summing)
            ("002", 0.1, 0.0),  # ambiguous on a safe slide is NOT negative
            ("002", 0.0, 0.0),  # neg
            ("043", 0.1, 0.0),  # excluded wins over ambiguous
        ],
        min_coverage=0.5,
    )
    assert out["label"].tolist() == [-1, 1, 1, -1, -1, 0, -1]
    assert out["reason"].tolist() == [
        "ambiguous_coverage",
        None,
        None,
        "ambiguous_coverage",
        "ambiguous_coverage",
        None,
        "excluded_slide",
    ]


def test_coverage_is_max_not_sum(table: pd.DataFrame) -> None:
    out = _label(table, [("001", 0.7, 0.6)])
    assert out["coverage"].tolist() == [0.7]


def test_all_tiles_kept(table: pd.DataFrame) -> None:
    rows = [("001", 0.0, 0.0), ("043", 0.0, 0.0), ("002", 0.0, 0.0)]
    out = _label(table, rows)
    assert len(out) == 3
    assert ((out["label"] == -1) == out["reason"].notna()).all()


@pytest.mark.parametrize("lg,hg", [(float("nan"), 0.0), (-0.1, 0.0), (0.0, 1.5)])
def test_invalid_coverage(table: pd.DataFrame, lg: float, hg: float) -> None:
    with pytest.raises(ValueError):
        _label(table, [("001", lg, hg)])


def test_bad_config() -> None:
    with pytest.raises(ValueError):
        LabelConfig(min_coverage=1.5)
    with pytest.raises(ValueError):
        LabelConfig(exclude_slides=frozenset({"43"}))


def test_summarize(table: pd.DataFrame) -> None:
    out = _label(
        table,
        [("001", 0.3, 0), ("001", 0, 0), ("002", 0, 0), ("002", 0, 0), ("003", 0, 0.2)],
    )
    s = summarize(out, by=["split", "fold"]).loc[("train", 0.0)]
    assert (s["pos"], s["neg"], s["pos_slides"], s["neg_slides"]) == (1, 2, 1, 1)
    assert s["neg_per_pos"] == 2.0
    assert s["not_safe_slide"] == 1


# --- multi-slide shards / test_final ------------------------------------


def _shards(tmp_path: Path) -> list[Path]:
    # Shards mix train, test_preliminary and test_final slides (like tiles_0000..0003).
    a = _tiles([("001", 0.3, 0), ("020", 0.9, 0), ("002", 0, 0)])
    b = _tiles([("020", 0, 0), ("010", 0, 0), ("003", 0, 0.1)]).assign(
        split="test_final"
    )
    a.to_parquet(tmp_path / "tiles_0000.parquet")
    b.drop(columns="split").to_parquet(tmp_path / "tiles_0001.parquet")
    return [tmp_path / "tiles_0000.parquet", tmp_path / "tiles_0001.parquet"]


def test_shards_drop_test_final_rows(table: pd.DataFrame, tmp_path: Path) -> None:
    out = load_tiles(
        None, table, ("train", "test_preliminary"), files=_shards(tmp_path)
    )
    assert "h020" not in set(out["slide_id"])
    assert sorted(out["name"]) == ["001", "002", "003", "010"]
    labelled = assign_labels(out, table)
    assert "020" not in set(labelled["name"])


def test_read_split_rows_filters_at_read(table: pd.DataFrame, tmp_path: Path) -> None:
    from ml.data.huncrc_labels import read_split_rows

    path = _shards(tmp_path)[0]
    df = read_split_rows(path, ["h001"], ["slide_id", LG])
    assert df["slide_id"].tolist() == ["h001"]


def test_per_slide_file_with_foreign_rows(table: pd.DataFrame, tmp_path: Path) -> None:
    for name in table["name"]:
        if name != "020":
            _tiles([(name, 0, 0)]).to_parquet(tmp_path / f"{name}.parquet")
    # a mislabelled per-slide file that also carries a test_final row
    _tiles([("001", 0, 0), ("020", 0, 0)]).to_parquet(tmp_path / "001.parquet")
    out = load_tiles(tmp_path, table, ("train",))
    assert "h020" not in set(out["slide_id"])


@pytest.mark.parametrize("splits", [("test_final",), ("train", "test_final")])
def test_test_final_refused(
    table: pd.DataFrame, tmp_path: Path, splits: tuple[str, ...]
) -> None:
    with pytest.raises(PermissionError):
        load_tiles(tmp_path, table, splits)
