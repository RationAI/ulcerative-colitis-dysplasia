"""HunCRC annotation squares -> exact per-tile coverage.

The annotation JSONs (``preprocessing/annotations/NNN.json``) hold axis-aligned
squares in level-0 px (1024 or 2048 px; patch-level labels of the TCIA ML
dataset), one item per (patch, class). Coverage of a tile is computed exactly
by coordinate compression + an integral image of the union of a class's
squares, so overlapping squares of one class are not double-counted.

Tile conventions (tiling_200: 224 px tiles, level 2, downsample 4):
``central_half`` = the 112 px centre, the convention of the tiling's
lowgrade_/highgrade_dysplasia columns; ``full`` = the whole 224 px tile, the
convention of tiling_200_allclasses' annot_<class> columns.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike


if TYPE_CHECKING:
    from pathlib import Path


CLASSES = (
    "lowgrade_dysplasia",
    "highgrade_dysplasia",
    "normal",
    "adenocarcinoma",
    "suspicious_for_invasion",
    "tumor_necrosis",
    "inflammation",
    "artifact",
    "resection_edge",
)
CARCINOMA_CLASSES = ("adenocarcinoma", "suspicious_for_invasion", "tumor_necrosis")


def load_squares(path: Path) -> dict[str, np.ndarray]:
    """``{class: (n, 4) array of x0, y0, x1, y1}`` in level-0 px; rejects non-rectangles."""
    items = json.loads(path.read_text())["items"]
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    for it in items:
        c = np.asarray(it["coordinates"], dtype=float)
        xs, ys = np.unique(c[:, 0]), np.unique(c[:, 1])
        if not (
            it["type"] == "polygon"
            and len(c) == 5
            and len(xs) == 2
            and len(ys) == 2
            and (c[0] == c[-1]).all()
        ):
            raise ValueError(f"{path.name}: item is not an axis-aligned rectangle")
        out.setdefault(it["name"], []).append((xs[0], ys[0], xs[1], ys[1]))
    return {k: np.array(v) for k, v in out.items()}


def rect_coverage(rects: np.ndarray | None, query: np.ndarray) -> np.ndarray:
    """Exact fraction of each query rect (n, 4) covered by the union of ``rects`` (m, 4)."""
    query = np.asarray(query, dtype=float)
    if rects is None or len(rects) == 0:
        return np.zeros(len(query))
    rects = np.asarray(rects, dtype=float)
    xs = np.unique(np.r_[rects[:, [0, 2]].ravel(), query[:, [0, 2]].ravel()])
    ys = np.unique(np.r_[rects[:, [1, 3]].ravel(), query[:, [1, 3]].ravel()])
    rx0, rx1 = np.searchsorted(xs, rects[:, 0]), np.searchsorted(xs, rects[:, 2])
    ry0, ry1 = np.searchsorted(ys, rects[:, 1]), np.searchsorted(ys, rects[:, 3])
    diff = np.zeros((len(ys) + 1, len(xs) + 1), dtype=np.int32)
    np.add.at(diff, (ry0, rx0), 1)
    np.add.at(diff, (ry0, rx1), -1)
    np.add.at(diff, (ry1, rx0), -1)
    np.add.at(diff, (ry1, rx1), 1)
    covered = (
        diff.cumsum(0).cumsum(1)[:-2, :-2] > 0
    )  # cell (j, i) spans ys[j:j+2], xs[i:i+2]
    area = np.outer(np.diff(ys), np.diff(xs)) * covered
    integral = np.zeros((len(ys), len(xs)))
    integral[1:, 1:] = area.cumsum(0).cumsum(1)
    qx0, qx1 = np.searchsorted(xs, query[:, 0]), np.searchsorted(xs, query[:, 2])
    qy0, qy1 = np.searchsorted(ys, query[:, 1]), np.searchsorted(ys, query[:, 3])
    a = (
        integral[qy1, qx1]
        - integral[qy0, qx1]
        - integral[qy1, qx0]
        + integral[qy0, qx0]
    )
    return a / ((query[:, 2] - query[:, 0]) * (query[:, 3] - query[:, 1]))


def tile_rects(
    x: ArrayLike, y: ArrayLike, region: str, downsample: float = 4.0, tile: int = 224
) -> np.ndarray:
    """Level-0 rects of tiles at level-``downsample`` coords; region 'full' or 'central_half'."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if region == "full":
        off, size = 0, tile
    elif region == "central_half":
        off, size = tile // 4, tile // 2
    else:
        raise ValueError(region)
    return np.stack(
        [(x + off) * downsample, (y + off) * downsample, (x + off + size) * downsample,
         (y + off + size) * downsample],
        axis=1,
    )  # fmt: skip


def tile_coverage(
    squares: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, region: str
) -> dict[str, np.ndarray]:
    """Coverage of every tile by every class in :data:`CLASSES`."""
    q = tile_rects(x, y, region)
    return {c: rect_coverage(squares.get(c), q) for c in CLASSES}


#: tolerance for "fully inside" on exact rational coverages
_FULL = 1 - 1e-9


def tile_flags(
    squares: dict[str, np.ndarray], x: ArrayLike, y: ArrayLike
) -> dict[str, np.ndarray]:
    """Step-7 tile definitions, from exact coverage.

    - ``normal``: central half fully inside the union of "normal" squares;
    - ``strict_normal``: normal and zero LG/HG coverage over the FULL tile;
    - ``carcinoma``: central half fully inside the union of the carcinoma
      squares (adenocarcinoma, suspicious_for_invasion, tumor_necrosis);
    - ``carcinoma_any``: central half touches a carcinoma square.
    """
    half, full = tile_rects(x, y, "central_half"), tile_rects(x, y, "full")
    carc = [squares[c] for c in CARCINOMA_CLASSES if c in squares]
    carc_rects = np.concatenate(carc) if carc else None
    lghg = [
        squares[c]
        for c in ("lowgrade_dysplasia", "highgrade_dysplasia")
        if c in squares
    ]
    lghg_rects = np.concatenate(lghg) if lghg else None
    normal = rect_coverage(squares.get("normal"), half) >= _FULL
    carc_cov = rect_coverage(carc_rects, half)
    return {
        "normal": normal,
        "strict_normal": normal & (rect_coverage(lghg_rects, full) == 0),
        "carcinoma": carc_cov >= _FULL,
        "carcinoma_any": carc_cov > 0,
    }
