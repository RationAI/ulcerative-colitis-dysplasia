"""Unit tests for ml.data.huncrc_features -- synthetic parquets only."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.data.huncrc_features import (
    EmbeddingMismatchError,
    build_features,
    cache_embedding,
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


def test_within_slide_cache_needs_annotations(emb_dir: Path, tmp_path: Path) -> None:
    cfg = LabelConfig(within_slide_negatives=True, exclude_carcinoma=True)
    with pytest.raises(ValueError, match="annotations"):
        build_features(emb_dir, _table(), ("train",), cfg, tmp_path, dim=DIM)


def test_within_slide_cache_from_synthetic_cohort(tmp_path: Path) -> None:
    from tests.huncrc_synth import NEW_RULES, build

    c = build(tmp_path)
    old = load_features(c["train"], expected_config=LabelConfig())
    new = load_features(c["train_within_slide"], expected_config=NEW_RULES)
    assert c["train_within_slide"] != c["train"]
    # positives unchanged (no carcinoma on train slides); extra negatives come only
    # from the strict-normal tiles of adenoma (positive) train slides
    assert new.meta["n_pos"] == old.meta["n_pos"]
    src = new.meta["n_neg_by_source"]
    assert src["safe_slide"] == old.meta["n_neg"]
    # 60 tiles per slide: strict normal = tiles 32..59 on each of 10 adenoma train slides
    assert src["positive_slide_normal"] == 10 * 28
    assert new.meta["annotation_flag_counts"]["carcinoma_and_lghg"] == 0


# --- embedding model / dimension --------------------------------------------


def _copy_as(emb_dir: Path, name: str, scale: float = 1.0, dim: int = DIM) -> Path:
    """The same tiles under embeddings/<name>, with other vectors (and dim)."""
    out = emb_dir.parent / "embeddings" / name
    out.mkdir(parents=True)
    for f in emb_dir.glob("*.parquet"):
        df = pd.read_parquet(f)
        df["embedding"] = [
            list(np.resize(np.asarray(v) * scale, dim)) for v in df["embedding"]
        ]
        df.to_parquet(out / f.name)
    return out


def test_default_embedding_keeps_the_existing_keys() -> None:
    t = _table()
    k = cache_key(LabelConfig(), ("train",), t["name"])
    assert k == cache_key(LabelConfig(), ("train",), t["name"],
                          model="prov-gigapath", dim=1536)  # fmt: skip
    assert "-e" not in k  # e.g. the share's mc0-train-r729e9556-v2-7a57c2cd
    v = cache_key(LabelConfig(), ("train",), t["name"], model="virchow2", dim=2560)
    assert "-evirchow2-d2560-" in v and v != k
    assert cache_key(LabelConfig(), ("train",), t["name"], model="prov-gigapath",
                     dim=2560) != k  # fmt: skip
    with pytest.raises(ValueError, match="embedding model"):
        cache_key(LabelConfig(), ("train",), t["name"], model="Virchow 2", dim=2560)


def test_cache_of_one_model_is_refused_for_the_other(
    emb_dir: Path, tmp_path: Path
) -> None:
    t = _table()
    gp = _copy_as(emb_dir, "prov-gigapath")
    v2 = _copy_as(emb_dir, "virchow2", scale=-1.0)  # same dim: only the model differs
    cache = tmp_path / "cache"
    a = build_features(gp, t, ("train",), LabelConfig(), cache, dim=DIM)
    b = build_features(v2, t, ("train",), LabelConfig(), cache, dim=DIM)
    assert a != b and "-eprov-gigapath-d8-" in a.name and "-evirchow2-d8-" in b.name
    fa, fb = load_features(a), load_features(b)
    assert fa.embedding == ("prov-gigapath", DIM) and fb.embedding == ("virchow2", DIM)
    assert np.array_equal(
        np.asarray(fb.features), -np.asarray(fa.features)
    )  # not reused
    load_features(a, expected_model="prov-gigapath", expected_dim=DIM)
    with pytest.raises(
        EmbeddingMismatchError, match=r"REFUSED.*prov-gigapath 8-d.*virchow2"
    ):
        load_features(a, expected_model="virchow2")
    with pytest.raises(
        EmbeddingMismatchError, match=r"REFUSED.*virchow2 8-d.*prov-gigapath"
    ):
        load_features(b, expected_model="prov-gigapath", expected_dim=DIM)
    with pytest.raises(EmbeddingMismatchError, match="expected prov-gigapath 1536-d"):
        load_features(a, expected_dim=1536)
    # a model name that is not the embeddings dir is refused before anything is built
    with pytest.raises(EmbeddingMismatchError, match="dir name must be the model name"):
        build_features(
            gp, t, ("train",), LabelConfig(), cache, dim=DIM, model="virchow2"
        )


def test_wrong_dim_refused_before_building(emb_dir: Path, tmp_path: Path) -> None:
    v2 = _copy_as(emb_dir, "virchow2", dim=2 * DIM)
    with pytest.raises(
        ValueError, match=rf"embedding length {2 * DIM} .* != dim {DIM}"
    ):
        build_features(v2, _table(), ("train",), LabelConfig(), tmp_path / "c", dim=DIM)
    assert not (tmp_path / "c").exists() or not any((tmp_path / "c").iterdir())


def test_old_cache_meta_names_its_embedding_by_source() -> None:
    old = {
        "key": "k",
        "dim": 1536,
        "source": "/p/preprocessing/embeddings/prov-gigapath",
    }
    assert cache_embedding(old) == ("prov-gigapath", 1536)
    assert cache_embedding({**old, "embedding_model": "virchow2", "dim": 2560}) == (
        "virchow2", 2560)  # fmt: skip


@pytest.mark.parametrize("arrow_type", ["float16", "float32"])
def test_low_precision_input_is_cast_to_float32(
    emb_dir: Path, tmp_path: Path, arrow_type: str
) -> None:
    """Virchow2 files store float16; the cache is float32 with the stored values."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    np_type = np.dtype(arrow_type)
    for f in emb_dir.glob("*.parquet"):
        t = pq.read_table(f)
        vecs = [np.asarray(v, dtype=np_type) for v in t.column("embedding").to_pylist()]
        col = pa.array(vecs, type=pa.list_(pa.from_numpy_dtype(np_type)))
        t = t.set_column(t.schema.get_field_index("embedding"), "embedding", col)
        pq.write_table(t, f)
    assert str(pq.read_schema(emb_dir / "001.parquet").field("embedding").type) == (
        f"list<element: {'halffloat' if arrow_type == 'float16' else 'float'}>"
    )
    fs = get_features(emb_dir, _table(), ("train",), LabelConfig(), tmp_path, dim=DIM)
    assert fs.features.dtype == np.float32
    want = np.asarray(_emb("001", 0), dtype=np_type).astype(np.float32)
    np.testing.assert_array_equal(fs.features[0], want)  # exact: no rounding twice


def test_non_float_embeddings_are_refused(emb_dir: Path, tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = emb_dir / "001.parquet"
    t = pq.read_table(f)
    col = pa.array([[1] * DIM] * t.num_rows, type=pa.list_(pa.int64()))
    pq.write_table(
        t.set_column(t.schema.get_field_index("embedding"), "embedding", col), f
    )
    with pytest.raises(TypeError, match="list of float16/float32/float64"):
        build_features(emb_dir, _table(), ("train",), LabelConfig(), tmp_path, dim=DIM)
