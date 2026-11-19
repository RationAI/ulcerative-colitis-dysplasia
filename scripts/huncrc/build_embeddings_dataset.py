"""Build per-split dataset directories for the HunCRC tile classifier.

``mlkit``'s ``SlidesTilesLoader`` accepts local ``paths``: a directory that
holds ``slides/`` and ``tiles/`` with (chunked) parquets. The HunCRC
embedding parquets already carry every tiling column plus ``embedding``, so
they are reused as the tiles table by symlink -- no vector data is copied.

Output layout, one directory per split::

    <out>/train/slides/slides.parquet
    <out>/train/tiles/001.parquet -> <embeddings>/001.parquet
    ...

Label rules implemented here (a temporary stand-in for IBD-45, see the
"HunCRC Tile Selection" card):

* A slide is a trusted source of NEGATIVES (``from_negative_slide``) only if
  it carries no LG/HG annotation at all, its slide-level ``CATEGORY`` is
  ``negative`` or ``non_neoplastic_lesion``, it is not a serrated lesion,
  and it is not a known conflict (043, 140, 135).
* Every other annotated slide is kept as a source of POSITIVES only; with
  ``negative_slides: true`` its unannotated tiles are dropped as unknown.
* Slides that are neither (adenoma/CRC slides with no LG/HG annotation, and
  the three conflicts) are dropped entirely.

Because the 16 slides carrying adenocarcinoma without any LG/HG annotation
fall in that last group, and carcinoma tiles on LG/HG slides are dropped as
unannotated, no carcinoma tile can end up labelled negative -- without
needing the per-class annotation files. Those are still required for the
per-class evaluation breakdown.

Example::

    python scripts/huncrc/build_embeddings_dataset.py \\
        --slide-csv /path/to/slide_level_annotations.csv \\
        --out /mnt/projects/huncrc_dysplasia/preprocessing/ml_dataset

    # five-slide smoke set for the overfitting sanity check
    python scripts/huncrc/build_embeddings_dataset.py \\
        --slide-csv /path/to/slide_level_annotations.csv \\
        --out /mnt/projects/huncrc_dysplasia/preprocessing/ml_dataset_smoke \\
        --max-slides 3 --max-negative-slides 2
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


#: Slides with a documented label conflict; see the HunCRC Tile Selection card.
#: 043: non_neoplastic_lesion carrying 137 LG cells. 140: non_neoplastic_lesion
#: with polyp type tubular_adenoma. 135: adenoma with polyp type
#: sessile_serrated and only normal cells.
EXCLUDED_SLIDES = frozenset({"043", "140", "135"})

#: Slide-level categories whose unannotated tissue may be used as negatives.
SAFE_CATEGORIES = frozenset({"negative", "non_neoplastic_lesion"})

SPLITS = ("train", "test_preliminary", "test_final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiling",
        type=Path,
        default=Path("/mnt/projects/huncrc_dysplasia/preprocessing/tiling_200"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            "/mnt/projects/huncrc_dysplasia/preprocessing/embeddings/prov-gigapath"
        ),
    )
    parser.add_argument("--slide-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-slides",
        type=int,
        default=None,
        help="Keep at most this many dysplasia slides per split (smoke sets).",
    )
    parser.add_argument(
        "--max-negative-slides",
        type=int,
        default=None,
        help="Keep at most this many negative-source slides per split.",
    )
    return parser.parse_args()


def slide_table(tiling: Path, slide_csv: Path) -> pd.DataFrame:
    """Slides with ``stem``, ``has_dysplasia``, category and the two flags."""
    slides = pd.read_parquet(tiling / "slides" / "slides.parquet")
    slides["stem"] = slides["path"].map(lambda p: Path(str(p)).name.split(".")[0])

    shards = sorted(glob.glob(str(tiling / "tiles" / "*.parquet")))
    if not shards:
        raise SystemExit(f"no tile parquets under {tiling / 'tiles'}")
    annotation = (
        pd.concat(
            [pd.read_parquet(f, columns=["slide_id", "annotation"]) for f in shards],
            ignore_index=True,
        )
        .groupby("slide_id")["annotation"]
        .max()
    )
    slides["has_dysplasia"] = slides["id"].map(annotation).fillna(0) > 0

    level = pd.read_csv(slide_csv)
    level["stem"] = level["slideID"].astype(str).str.zfill(3)
    missing = set(slides["stem"]) - set(level["stem"])
    if missing:
        raise SystemExit(f"{len(missing)} slides are not in {slide_csv}: {sorted(missing)[:5]}")

    slides = slides.merge(
        level[["stem", "CATEGORY", "Polyp type"]], on="stem", how="left"
    )
    serrated = (
        slides["Polyp type"].astype(str).str.contains("serrated", case=False, na=False)
    )
    excluded = slides["stem"].isin(EXCLUDED_SLIDES)

    slides["from_negative_slide"] = (
        ~slides["has_dysplasia"]
        & slides["CATEGORY"].isin(SAFE_CATEGORIES)
        & ~serrated
        & ~excluded
    )
    slides["keep"] = (slides["has_dysplasia"] | slides["from_negative_slide"]) & ~excluded
    return slides


def build_split(
    slides: pd.DataFrame,
    split: str,
    embeddings: Path,
    out: Path,
    max_slides: int | None,
    max_negative_slides: int | None,
) -> pd.DataFrame:
    kept = slides[slides["keep"] & (slides["split"] == split)].sort_values("stem")

    if max_slides is not None or max_negative_slides is not None:
        positives = kept[~kept["from_negative_slide"]]
        negatives = kept[kept["from_negative_slide"]]
        if max_slides is not None:
            positives = positives.head(max_slides)
        if max_negative_slides is not None:
            negatives = negatives.head(max_negative_slides)
        kept = pd.concat([positives, negatives]).sort_values("stem")

    available = []
    for stem in kept["stem"]:
        if (embeddings / f"{stem}.parquet").is_file():
            available.append(stem)
        else:
            print(f"  [{split}] {stem}: no embedding parquet, skipped")
    kept = kept[kept["stem"].isin(available)].reset_index(drop=True)
    if kept.empty:
        print(f"  [{split}] nothing to write")
        return kept

    split_dir = out / split
    (split_dir / "slides").mkdir(parents=True, exist_ok=True)
    tiles_dir = split_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    #: Drop stale symlinks so a rebuild with fewer slides does not leave
    #: yesterday's slides in the tiles table.
    for stale in tiles_dir.glob("*.parquet"):
        stale.unlink()

    for stem in kept["stem"]:
        (tiles_dir / f"{stem}.parquet").symlink_to(
            (embeddings / f"{stem}.parquet").resolve()
        )

    kept.drop(columns=["keep"]).to_parquet(
        split_dir / "slides" / "slides.parquet", index=False
    )
    return kept


def main() -> None:
    args = parse_args()
    slides = slide_table(args.tiling, args.slide_csv)

    print(f"slides:              {len(slides)}")
    print(f"  with LG/HG:        {int(slides.has_dysplasia.sum())}")
    print(f"  negative sources:  {int(slides.from_negative_slide.sum())}")
    print(f"  kept:              {int(slides.keep.sum())}")
    print(f"  dropped:           {int((~slides.keep).sum())}")
    print("\ndropped slides:")
    print(
        slides[~slides["keep"]][["stem", "split", "CATEGORY", "Polyp type"]]
        .sort_values(["CATEGORY", "stem"])
        .to_string(index=False)
    )
    print("\nnegative sources by polyp type:")
    print(
        pd.crosstab(
            slides[slides["from_negative_slide"]]["split"],
            slides[slides["from_negative_slide"]]["Polyp type"],
        )
    )

    print()
    manifests = []
    for split in SPLITS:
        kept = build_split(
            slides,
            split,
            args.embeddings,
            args.out,
            args.max_slides,
            args.max_negative_slides,
        )
        if kept.empty:
            continue
        positives = int((~kept["from_negative_slide"]).sum())
        negatives = int(kept["from_negative_slide"].sum())
        print(
            f"{split:17s} {len(kept):4d} slides "
            f"({positives} positive sources, {negatives} negative sources) "
            f"-> {args.out / split}"
        )
        kept = kept.assign(dataset_split=split)
        manifests.append(kept)

    if manifests:
        args.out.mkdir(parents=True, exist_ok=True)
        pd.concat(manifests, ignore_index=True)[
            [
                "stem",
                "id",
                "dataset_split",
                "fold",
                "CATEGORY",
                "Polyp type",
                "has_dysplasia",
                "from_negative_slide",
            ]
        ].to_csv(args.out / "dataset_manifest.csv", index=False)
        print(f"\nmanifest: {args.out / 'dataset_manifest.csv'}")


if __name__ == "__main__":
    main()
