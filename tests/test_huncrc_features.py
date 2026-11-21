"""Unit tests for ml.data.huncrc_features -- synthetic parquets only."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.data.huncrc_features import (
    build_features,
    cache_key,
    get_features,
    load_features,
)
from ml.data.huncrc_labels import HG, LG, LabelConfig, load_slide_table


DIM = 8


def _table() -> pd.DataFrame:
    rows = [
        ("001", "train", 0.0, "adenoma", "tubular_adenoma"),
        ("002", "train", 1.0, "negative", "Not_applicable"),
        ("003", "train", 1.0, "CRC", "Not_applicable"),
        ("043", "train", 0.0, "non_neoplastic_lesion", "Not_applicable"),
        ("010", "test_preliminary", None, "non_neoplastic_lesion", "hyperplastic"),
        ("020", "test_final", None, "negative", "Not_applicable"),
    ]
    slides = pd.DataFrame(
        {
            "id": [f"h{r[0]}" for r in rows],
            "path": [f"/d/{r[0]}.mrxs" for r in rows],
            "split": [r[1] for r in rows],
            "fold": [r[2] for r in rows],
        }
    )
    csv = pd.DataFrame(
        {
            "slideID": [r[0] for r in rows],
            "CATEGORY": [r[3] for r in rows],
            "Polyp type": [r[4] for r in rows],
        },
        dtype=str,
    )
    return load_slide_table(slides, csv)


#: per slide: (LG, HG) per tile
COVER: dict[str, list[tuple[float, float]]] = {
    "001": [(0.4, 0), (0, 0), (0.0, 0.2), (0, 0), (0.1, 0)],  # 3 pos, 2 not_safe
    "002": [(0, 0)] * 4,  # 4 neg
    "003": [(0, 0), (0, 0.9)],  # 1 pos, 1 not_safe
    "043": [(0.5, 0), (0, 0)],  # excluded
    "010": [(0, 0)] * 3,  # 3 neg (test_preliminary)
}


def _emb(name: str, i: int) -> list[float]:
    # deterministic, float64 values that are not exactly representable in float32
    base = int(name) * 100 + i
    return [base + j / 3 for j in range(DIM)]


@pytest.fixture
def emb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "emb"
    d.mkdir()
    for name, cover in COVER.items():
        pd.DataFrame(
            {
                "slide_id": f"h{name}",
                "x": np.arange(len(cover)) * 112,
                "y": 0,
                LG: [c[0] for c in cover],
                HG: [c[1] for c in cover],
                "fold": 9.0,  # bogus tile-side fold: must never be used
                "embedding": [_emb(name, i) for i in range(len(cover))],
            }
        ).to_parquet(d / f"{name}.parquet", row_group_size=2)
    # no 020.parquet: reading test_final would raise FileNotFoundError
    return d


def test_build_and_load(emb_dir: Path, tmp_path: Path) -> None:
    t = _table()
    fs = get_features(
        emb_dir, t, ("train",), LabelConfig(), tmp_path / "cache", dim=DIM, batch_size=2
    )
    assert fs.features.dtype == np.float32
    assert isinstance(fs.features, np.memmap)
    assert fs.features.shape == (3 + 4 + 1, DIM)
    assert fs.labels.tolist() == [1, 1, 1, 0, 0, 0, 0, 1]
    assert fs.slide_names.tolist() == ["001"] * 3 + ["002"] * 4 + ["003"]
    assert fs.folds.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]  # from slides.parquet, not 9
    assert fs.x.tolist() == [0, 224, 448, 0, 112, 224, 336, 112]
    expected = np.array(
        [_emb("001", i) for i in (0, 2, 4)]
        + [_emb("002", i) for i in range(4)]
        + [_emb("003", 1)],
        dtype=np.float64,
    ).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(fs.features), expected)
    assert fs.meta["n_pos"] == 4 and fs.meta["n_neg"] == 4
    assert "043" not in set(fs.slide_names)


def test_test_preliminary_folds_are_minus_one(emb_dir: Path, tmp_path: Path) -> None:
    fs = get_features(
        emb_dir, _table(), ("test_preliminary",), LabelConfig(), tmp_path, dim=DIM
    )
    assert fs.folds.tolist() == [-1, -1, -1]
    assert fs.labels.tolist() == [0, 0, 0]


def test_test_final_refused(emb_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        build_features(
            emb_dir, _table(), ("train", "test_final"), LabelConfig(), tmp_path, dim=DIM
        )


def test_cache_reused_and_keyed_by_min_coverage(emb_dir: Path, tmp_path: Path) -> None:
    t = _table()
    a = build_features(emb_dir, t, ("train",), LabelConfig(), tmp_path, dim=DIM)
    for f in emb_dir.glob("*.parquet"):
        f.unlink()  # a rebuild would now fail
    assert build_features(emb_dir, t, ("train",), LabelConfig(), tmp_path, dim=DIM) == a
    assert load_features(a).meta["n"] == 8
    assert a.name.startswith("mc0-")
    k = cache_key(LabelConfig(min_coverage=0.25), ("train",), t["name"])
    assert k.startswith("mc0.25-") and k != a.name


def test_min_coverage_drops_ambiguous(emb_dir: Path, tmp_path: Path) -> None:
    fs = get_features(
        emb_dir, _table(), ("train",), LabelConfig(min_coverage=0.3), tmp_path, dim=DIM
    )
    # 001: 0.4 pos, 0.2 & 0.1 ambiguous; 003: 0.9 pos
    assert fs.labels.tolist() == [1, 0, 0, 0, 0, 1]
    assert fs.slide_names.tolist() == ["001", "002", "002", "002", "002", "003"]


def test_wrong_dim_raises(emb_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="embedding length"):
        build_features(
            emb_dir, _table(), ("train",), LabelConfig(), tmp_path, dim=DIM + 1
        )
    assert not any(p.name.startswith("mc") for p in tmp_path.iterdir())


def test_foreign_rows_in_slide_file_raise(emb_dir: Path, tmp_path: Path) -> None:
    df = pd.read_parquet(emb_dir / "001.parquet")
    df.loc[1, "slide_id"] = "h020"
    df.to_parquet(emb_dir / "001.parquet")
    with pytest.raises(ValueError, match="other slides"):
        build_features(emb_dir, _table(), ("train",), LabelConfig(), tmp_path, dim=DIM)


def test_nan_embedding_raises(emb_dir: Path, tmp_path: Path) -> None:
    df = pd.read_parquet(emb_dir / "002.parquet")
    df.at[0, "embedding"] = [float("nan")] * DIM
    df.to_parquet(emb_dir / "002.parquet")
    with pytest.raises(ValueError, match="non-finite"):
        build_features(emb_dir, _table(), ("train",), LabelConfig(), tmp_path, dim=DIM)


# --- cache key / stale-cache protection ------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"min_coverage": 0.1},
        {"exclude_slides": frozenset({"043", "140"})},
        {"safe_categories": frozenset({"negative"})},
        {"exclude_serrated": False},
    ],
)
def test_rule_parameters_change_fingerprint_and_key(change: dict[str, Any]) -> None:
    from ml.data.huncrc_labels import rules_fingerprint

    t = _table()
    base, other = LabelConfig(), LabelConfig(**change)
    assert rules_fingerprint(base) != rules_fingerprint(other)
    assert cache_key(base, ("train",), t["name"]) != cache_key(
        other, ("train",), t["name"]
    )


def test_fingerprint_ignores_set_order() -> None:
    from ml.data.huncrc_labels import rules_fingerprint

    a = LabelConfig(exclude_slides=frozenset({"043", "140", "135"}))
    b = LabelConfig(exclude_slides=frozenset(["135", "043", "140"]))
    assert rules_fingerprint(a) == rules_fingerprint(b)


def test_versions_change_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.data import huncrc_features, huncrc_labels

    t = _table()
    k0 = cache_key(LabelConfig(), ("train",), t["name"])
    monkeypatch.setattr(huncrc_labels, "LABEL_RULES_VERSION", 999)
    k1 = cache_key(LabelConfig(), ("train",), t["name"])
    monkeypatch.setattr(huncrc_features, "CACHE_VERSION", 999)
    k2 = cache_key(LabelConfig(), ("train",), t["name"])
    assert len({k0, k1, k2}) == 3
    assert "-r" in k0 and "-v" in k0


def test_changed_rule_does_not_reuse_cache(emb_dir: Path, tmp_path: Path) -> None:
    t = _table()
    a = build_features(emb_dir, t, ("train",), LabelConfig(), tmp_path, dim=DIM)
    for f in emb_dir.glob("*.parquet"):
        f.unlink()
    # same splits/slides, different rule -> must rebuild (and fail: sources gone)
    with pytest.raises(FileNotFoundError):
        build_features(
            emb_dir,
            t,
            ("train",),
            LabelConfig(exclude_slides=frozenset()),
            tmp_path,
            dim=DIM,
        )
    assert build_features(emb_dir, t, ("train",), LabelConfig(), tmp_path, dim=DIM) == a


def test_load_with_wrong_expected_config_raises(emb_dir: Path, tmp_path: Path) -> None:
    from ml.data.huncrc_features import StaleCacheError

    d = build_features(emb_dir, _table(), ("train",), LabelConfig(), tmp_path, dim=DIM)
    load_features(d, expected_config=LabelConfig())
    with pytest.raises(StaleCacheError):
        load_features(d, expected_config=LabelConfig(min_coverage=0.5))


def test_renamed_or_old_cache_is_refused(emb_dir: Path, tmp_path: Path) -> None:
    import json

    from ml.data.huncrc_features import StaleCacheError

    t = _table()
    d = build_features(emb_dir, t, ("train",), LabelConfig(), tmp_path, dim=DIM)
    meta = json.loads((d / "meta.json").read_text())
    # (1) an older cache without version / fingerprint fields
    old = {
        k: v
        for k, v in meta.items()
        if k not in ("label_rules_version", "rules_fingerprint")
    }
    old["version"] = 1
    (d / "meta.json").write_text(json.dumps(old))
    with pytest.raises(StaleCacheError):
        load_features(d)
    # (2) right versions, but built with another rule and renamed into this key
    meta["rules_fingerprint"] = "deadbeef"
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(StaleCacheError):
        build_features(emb_dir, t, ("train",), LabelConfig(), tmp_path, dim=DIM)


# --- all-tile evaluation caches --------------------------------------------


def test_all_tile_cache_keeps_every_tile(emb_dir: Path, tmp_path: Path) -> None:
    t = _table()
    d = build_features(
        emb_dir, t, ("train",), LabelConfig(), tmp_path, dim=DIM, mode="all"
    )
    assert "-all-" in d.name
    fs = load_features(d, expected_config=LabelConfig(), mode="all")
    n_all = sum(len(COVER[n]) for n in ("001", "002", "003", "043"))
    assert len(fs) == n_all and fs.features.shape == (n_all, DIM)
    # 001: 3 pos + 2 not_safe; 002: 4 neg; 003: 1 pos + 1 not_safe; 043: 2 excluded
    assert fs.labels.tolist() == [1, -1, 1, -1, 1, 0, 0, 0, 0, -1, 1, -1, -1]
    assert fs.reasons is not None
    assert fs.reasons.tolist() == (
        ["", "not_safe_slide", "", "not_safe_slide", ""]
        + [""] * 4
        + ["not_safe_slide", ""]
        + ["excluded_slide"] * 2
    )
    assert fs.reasons is not None and fs.lg is not None and fs.hg is not None
    np.testing.assert_allclose(fs.lg[:5], [0.4, 0, 0, 0, 0.1])
    np.testing.assert_allclose(fs.hg[:5], [0, 0, 0.2, 0, 0])
    # the labelled rows are the same vectors as in the labelled cache
    lab = get_features(emb_dir, t, ("train",), LabelConfig(), tmp_path, dim=DIM)
    np.testing.assert_array_equal(np.asarray(fs.features)[fs.labels >= 0], lab.features)


def test_all_tile_cache_refused_by_default_loader(
    emb_dir: Path, tmp_path: Path
) -> None:
    d = build_features(
        emb_dir, _table(), ("train",), LabelConfig(), tmp_path, dim=DIM, mode="all"
    )
    with pytest.raises(ValueError, match="'all'-tile cache"):
        load_features(d)
    lab = build_features(
        emb_dir, _table(), ("train",), LabelConfig(), tmp_path, dim=DIM
    )
    assert lab != d
    with pytest.raises(ValueError, match="'labelled'-tile cache"):
        load_features(lab, mode="all")


def test_all_tile_cache_still_refuses_test_final(emb_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        build_features(
            emb_dir,
            _table(),
            ("test_final",),
            LabelConfig(),
            tmp_path,
            dim=DIM,
            mode="all",
        )


def test_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        cache_key(LabelConfig(), ("train",), ["001"], mode="some")
