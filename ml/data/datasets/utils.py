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
    from_negative_slide: bool | None = None,
    annotation_column: str = "annotation",
) -> HFDataset:
    """Restrict which negative tiles are kept when ``negative_slides`` is set.

    A tile counts as negative when it has no annotation
    (``annotation_column`` is ``0``). In ``negative_slides`` mode, negatives
    are kept only from slides that are a trusted source of negatives;
    negatives from annotated slides are treated as unknown and dropped.
    Positive tiles are always kept regardless of origin.

    ``from_negative_slide`` is the slide's flag. Pass ``None`` (the default)
    for a tiling that carries a per-tile ``from_negative_slide`` column, as
    the IKEM tile filtering writes it. Pass the slide's boolean when the flag
    lives in the slide metadata instead -- the HunCRC embedding parquets are
    reused as the tiles table and cannot carry an extra column, so there the
    flag is written into ``slides.parquet``.
    """
    if not negative_slides:
        return tiles

    if from_negative_slide is None:

        def keep(tile: dict[str, Any]) -> bool:
            is_negative = tile[annotation_column] == 0
            return (not is_negative) or tile["from_negative_slide"]

        return tiles.filter(keep)

    if from_negative_slide:
        return tiles

    return tiles.filter(lambda tile: tile[annotation_column] != 0)
