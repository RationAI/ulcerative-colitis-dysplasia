"""Synthetic tests for ml.data.huncrc_annotations."""

import json
from pathlib import Path

import numpy as np
import pytest

from ml.data.huncrc_annotations import (
    load_squares,
    rect_coverage,
    tile_coverage,
    tile_rects,
)


def _brute(rects: np.ndarray, q: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Pixel-centre sampling on a fine grid (integer-aligned inputs -> exact)."""
    out = []
    for x0, y0, x1, y1 in q:
        xs = np.arange(x0, x1, step) + step / 2
        ys = np.arange(y0, y1, step) + step / 2
        X, Y = np.meshgrid(xs, ys)
        inside = np.zeros(X.shape, bool)
        for a0, b0, a1, b1 in rects:
            inside |= (a0 <= X) & (a1 > X) & (b0 <= Y) & (b1 > Y)
        out.append(inside.mean())
    return np.array(out)


def test_rect_coverage_matches_brute_force_with_overlaps() -> None:
    rng = np.random.default_rng(0)
    rects = rng.integers(0, 40, (12, 2))
    rects = np.c_[rects, rects + rng.integers(1, 15, (12, 2))].astype(float)
    q = rng.integers(0, 45, (30, 2))
    q = np.c_[q, q + rng.integers(1, 10, (30, 2))].astype(float)
    np.testing.assert_allclose(rect_coverage(rects, q), _brute(rects, q), atol=1e-12)


def test_overlapping_squares_not_double_counted() -> None:
    rects = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [5, 0, 15, 10]], float)
    assert rect_coverage(rects, np.array([[0, 0, 20, 10]], float))[0] == pytest.approx(
        0.75
    )


def test_empty_and_disjoint() -> None:
    q = np.array([[0, 0, 4, 4]], float)
    assert rect_coverage(None, q)[0] == 0.0
    assert rect_coverage(np.array([[10, 10, 20, 20]], float), q)[0] == 0.0


def test_tile_rects_conventions() -> None:
    full = tile_rects([100], [200], "full")
    half = tile_rects([100], [200], "central_half")
    np.testing.assert_array_equal(full, [[400, 800, 1296, 1696]])
    np.testing.assert_array_equal(half, [[624, 1024, 1072, 1472]])
    with pytest.raises(ValueError):
        tile_rects([0], [0], "ring")


def test_centre_inside_square_but_ring_touches_neighbour(tmp_path: Path) -> None:
    # a normal square fully containing the tile centre, a LG square touching only the ring
    items = [
        {"name": "normal", "type": "polygon",
         "coordinates": [[600, 1000], [1100, 1000], [1100, 1500], [600, 1500], [600, 1000]]},
        {"name": "lowgrade_dysplasia", "type": "polygon",
         "coordinates": [[1200, 800], [1400, 800], [1400, 1700], [1200, 1700], [1200, 800]]},
    ]  # fmt: skip
    p = tmp_path / "001.json"
    p.write_text(json.dumps({"items": items}))
    sq = load_squares(p)
    half = tile_coverage(sq, np.array([100]), np.array([200]), "central_half")
    full = tile_coverage(sq, np.array([100]), np.array([200]), "full")
    assert half["normal"][0] == 1.0 and half["lowgrade_dysplasia"][0] == 0.0
    assert full["lowgrade_dysplasia"][0] > 0.0  # -> normal but not strict normal


def test_load_squares_rejects_non_rectangles(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"items": [{"name": "normal", "type": "polygon",
                 "coordinates": [[0, 0], [5, 0], [0, 5], [0, 0]]}]}))  # fmt: skip
    with pytest.raises(ValueError, match="axis-aligned"):
        load_squares(p)


def test_tile_flags_definitions() -> None:
    from ml.data.huncrc_annotations import tile_flags

    def sq(x0: float, x1: float) -> np.ndarray:
        return np.array([[x0, -100, x1, 1000]], float)

    squares = {
        "normal": sq(0, 448 * 6),  # covers centres of tiles 0..4
        "lowgrade_dysplasia": sq(448 * 6, 448 * 7),  # touches the full extent of tile 5
        "adenocarcinoma": sq(448 * 8, 448 * 10),
    }
    x = np.arange(10) * 112
    f = tile_flags(squares, x, np.zeros(10))
    assert f["normal"].tolist()[:6] == [True] * 5 + [False]
    # tile 5's full extent [2240, 3136] overlaps LG [2688, 3136]; tiles <= 4 do not
    assert f["strict_normal"].tolist()[:6] == [True] * 5 + [False]
    assert f["carcinoma"].tolist()[8] and not f["carcinoma"].tolist()[7]
    assert f["carcinoma_any"].tolist()[7] and not f["carcinoma_any"].tolist()[6]
