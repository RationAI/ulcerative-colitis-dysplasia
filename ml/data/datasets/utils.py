from typing import Any

from datasets import Dataset as HFDataset


def filter_tiles(
    tiles: HFDataset,
    thresholds: dict[str, float] | None = None,
    min_thresholds: dict[str, float] | None = None,
) -> HFDataset:
    """Filter tiles by per-column value thresholds.

    ``thresholds`` (upper bounds) keep tiles where ``column <= value``, e.g.
    QC columns like blur/artifacts must be low. ``min_thresholds`` (lower
    bounds) keep tiles where ``column >= value``, e.g. epithelium coverage
    must be high. A tile is kept only if it satisfies every bound.
    """
    thresholds = thresholds or {}
    min_thresholds = min_thresholds or {}

    def keep(tile: dict[str, Any]) -> bool:
        return all(tile[col] <= thr for col, thr in thresholds.items()) and all(
            tile[col] >= thr for col, thr in min_thresholds.items()
        )

    return tiles.filter(keep)


def filter_negative_origin(
    tiles: HFDataset,
    negative_slides: bool = False,
) -> HFDataset:
    """Restrict which negative tiles are kept when ``negative_slides`` is set.

    A tile counts as negative when it has no annotation (``annotation`` is
    ``0``). In ``negative_slides`` mode, negatives are kept only from wholly
    negative slides (``from_negative_slide`` is true); negatives from
    annotated slides are treated as unknown and dropped. Positive tiles are
    always kept regardless of origin.
    """
    if not negative_slides:
        return tiles

    def keep(tile: dict[str, Any]) -> bool:
        is_negative = tile["annotation"] == 0
        return (not is_negative) or tile["from_negative_slide"]

    return tiles.filter(keep)
