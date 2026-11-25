"""Convert hunCRC per-tile labels into EMPAIA JSONs that preprocessing/tiling.py reads.

    uv run python scripts/make_huncrc_annotations.py
    uv run python scripts/make_huncrc_annotations.py --slides 001 002 --out /tmp/annot
    uv run python scripts/make_huncrc_annotations.py --display-names

Why this exists: tiling takes vectors out of one EMPAIA JSON per slide
(preprocessing/tiling.py:87,124 -- annot_folder / f"{slide_name}.json") and hunCRC ships
no vector annotations at all. slide_level_annotations.csv is slide-level only -- 200 rows
of CATEGORY / Haggit-level / Biopsy-or-polyp / Polyp type, no geometry. The only geometry
in the dataset is the per-tile label tables:

    ML_dataset/<zoom_part>/<NNN>/<NNN>_labels.csv
        topleft_x, topleft_y    level-0 slide pixels
        <class>                 0/1, one column per class

so each labeled tile becomes one axis-aligned square polygon, which is exactly what that
table asserts: that whole tile is that class.

Coordinate space needs NO conversion. tile_annotations scales the tile ROI UP by
downsample before intersecting (ratiopath/tiling/annotations.py:65), i.e. it compares in
level-0 px, which is the space topleft_x/topleft_y are already in.

Square side is READ PER CSV, not a constant. Measured on slide 001: zoom_1 pitch is 1024
level-0 px and zoom_2's is 2048, and get_tile_df (preprocessing/create_dataset.py:66-73)
prefers zoom_1 -- so a hardcoded 2048, which is what scripts/make_dysplasia_masks.py's
ZOOM_PITCH assumes, would draw every zoom_1 polygon at 2x true area. The pitch comes from
the diffs of the sorted unique topleft values, falling back to --pitch when a slide has
too few tiles to measure.

EMPIA name <-> target_groups. tiling matches class names with rf"^{re.escape(label)}$"
against annotation["name"] (tiling.py:90, ratiopath/parsers/empaia_parser.py:42-46), so a
mismatch yields empty polygons SILENTLY -- the job succeeds and every tile is labeled
background. configs/preprocessing/tiling.yaml:9-11 says "LG Dysplasia"/"HG Dysplasia";
the CSV columns are lowgrade_dysplasia/highgrade_dysplasia. Default here is the raw column
names, all ten classes, lossless -- then pass the matching groups to tiling:

    target_groups=[lowgrade_dysplasia,highgrade_dysplasia]

--display-names instead renames the two dysplasia columns to the IBD spelling, so
tiling.yaml works untouched. Prefer the flag over editing tiling.yaml: this branch is
hunCRC-only and tiling.yaml is shared with IBD, whose slides do use those names.

Writes <out>/<NNN>.json. Nothing else is touched. Tiling reads annot_path OUT OF
dataset.csv (preprocessing/tiling.py:223 -> add_annot_path -> df.loc[stem, "annot_path"])
and the hunCRC create_dataset selects only [case_id, slide_path, tiles_path]
(preprocessing/create_dataset.py:106), where IBD's version carried annot_path
(master:create_dataset.py:82-85). So dataset.csv needs that column before tiling can
consume these JSONs -- rerun create_dataset with it added, then split, whose CSVs tiling
also reads (preprocessing/tiling.py:308).
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import pandas as pd


#: Same glob create_dataset uses, so this script and dataset.csv agree on which slides
#: exist and which zoom part a slide's labels came from.
TILES_GLOB = "*/[0-9][0-9][0-9]/[0-9][0-9][0-9]_labels.csv"
TILES_PATH = Path(
    "/mnt/data/Public/Public/colorectum/TCIA_HunCRC"
    "/PKG - Hungarian-Colorectal-Screening_ML-dataset/ML_dataset"
)
OUT_DIR = Path("/mnt/projects/huncrc_dysplasia/annotations")

#: CSV column -> EMPAIA annotation name. Only the two dysplasia classes differ under
#: --display-names; everything else keeps the dataset's own spelling.
DISPLAY_NAMES = {
    "lowgrade_dysplasia": "LG Dysplasia",
    "highgrade_dysplasia": "HG Dysplasia",
}
COORD_COLUMNS = ("topleft_x", "topleft_y")
#: burnt_out_pct / low_saturation_pct are QA floats and n_masks_for_slide is a count, so
#: class columns are inferred as the int 0/1 remainder rather than hardcoded.
NON_CLASS_COLUMNS = frozenset(
    (*COORD_COLUMNS, "fname", "burnt_out_pct", "low_saturation_pct", "n_masks_for_slide")
)


def class_columns(df: pd.DataFrame) -> list[str]:
    """Binary 0/1 columns, i.e. the label classes, in file order."""
    out = []
    for col in df.columns:
        if col in NON_CLASS_COLUMNS:
            continue
        values = df[col].dropna().unique()
        if set(values) <= {0, 1}:
            out.append(col)
    return out


def measure_pitch(df: pd.DataFrame, axis: str, fallback: int | None) -> int:
    """Smallest gap between distinct tile origins on one axis.

    Both axes are measured by the caller and the min is taken, so a slide labeled on only
    one row or column cannot report a bogus pitch from the other.
    """
    values = sorted({int(v) for v in df[axis]})
    if len(values) < 2:
        if fallback is None:
            raise ValueError(
                f"only {len(values)} distinct {axis} value(s); cannot measure pitch. "
                "Pass --pitch."
            )
        return fallback
    return min(b - a for a, b in pairwise(values))


def label_pitch(df: pd.DataFrame, fallback: int | None) -> int:
    return min(
        measure_pitch(df, "topleft_x", fallback),
        measure_pitch(df, "topleft_y", fallback),
    )


def square(x: int, y: int, side: int) -> list[list[float]]:
    """Closed ring for one tile, in level-0 px. EMPAIA wants the first point repeated."""
    x2, y2 = x + side, y + side
    return [[float(x), float(y)], [float(x2), float(y)],
            [float(x2), float(y2)], [float(x), float(y2)], [float(x), float(y)]]


def build_annotations(
    df: pd.DataFrame, pitch: int, names: dict[str, str]
) -> tuple[dict, dict[str, int]]:
    items: list[dict] = []
    counts: dict[str, int] = {}
    for column in class_columns(df):
        name = names.get(column, column)
        hits = df.loc[df[column] == 1, list(COORD_COLUMNS)]
        for x, y in hits.itertuples(index=False):
            items.append(
                {
                    "name": name,
                    "type": "polygon",
                    "coordinates": square(int(x), int(y), pitch),
                }
            )
        counts[name] = counts.get(name, 0) + len(hits)
    #: EMPAIA's only required structure is "items" (empaia_parser.py:42,66); the rest is
    #: metadata this deployment ignores, so nothing invented goes in it.
    return {"items": items}, counts


def label_files(tiles_path: Path, slides: list[str] | None) -> dict[str, Path]:
    """<slide_id> -> <NNN>_labels.csv, zoom_1 winning over zoom_2.

    Mirrors get_tile_df (preprocessing/create_dataset.py:66-73) exactly: sorted glob for
    determinism, priority 0 for any zoom_1_* part, 1 otherwise, lower wins.
    """
    best: dict[str, tuple[int, Path]] = {}
    for path in sorted(tiles_path.glob(TILES_GLOB)):
        slide_id = path.stem.removesuffix("_labels")
        if slides is not None and slide_id not in slides:
            continue
        part = path.parent.parent.name
        priority = 0 if part.startswith("zoom_1") else 1
        if slide_id not in best or priority < best[slide_id][0]:
            best[slide_id] = (priority, path)
    return {slide_id: path for slide_id, (_, path) in best.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles-path", default=str(TILES_PATH), type=Path)
    parser.add_argument("--out", default=str(OUT_DIR), type=Path)
    parser.add_argument(
        "--slides", nargs="*", default=None,
        help="3-digit ids to convert; default is every slide with labels",
    )
    parser.add_argument(
        "--pitch", type=int, default=None,
        help="tile side in level-0 px, only for slides too small to measure (see docstring)",
    )
    parser.add_argument(
        "--display-names", action="store_true",
        help='name the two dysplasia classes "LG/HG Dysplasia" to match tiling.yaml',
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rewrite JSONs that already exist"
    )
    args = parser.parse_args()

    tiles_path: Path = args.tiles_path
    out_dir: Path = args.out
    if not tiles_path.is_dir():
        sys.stderr.write(
            f"{tiles_path} not found. It is a /mnt/data path, so run this where the "
            "public cloud's rationai/Data is mounted (this notebook, or a kube-job).\n"
        )
        return 1

    files = label_files(tiles_path, args.slides)
    if not files:
        sys.stderr.write(f"No {TILES_GLOB} matches under {tiles_path}\n")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    names = DISPLAY_NAMES if args.display_names else {}

    written = skipped = 0
    total_items = 0
    per_class: dict[str, int] = {}
    for slide_id, labels_path in sorted(files.items()):
        dest = out_dir / f"{slide_id}.json"
        if dest.exists() and not args.overwrite:
            skipped += 1
            continue

        df = pd.read_csv(labels_path)
        if df[list(COORD_COLUMNS)].isna().any().any():
            sys.stderr.write(f"{labels_path}: NaN in topleft_x/y, skipping\n")
            skipped += 1
            continue

        try:
            pitch = label_pitch(df, args.pitch)
        except ValueError as e:
            sys.stderr.write(f"{labels_path}: {e}\n")
            skipped += 1
            continue

        payload, counts = build_annotations(df, pitch, names)
        #: An empty items list is a legitimate NEGATIVE-equivalent (no labeled tile),
        #: and tiling only opens the file when annot_path != "NEGATIVE", so write it.
        with open(dest, "w") as handle:
            json.dump(payload, handle)
        written += 1
        total_items += len(payload["items"])
        for name, count in counts.items():
            per_class[name] = per_class.get(name, 0) + count

    print(f"wrote {written} JSONs to {out_dir} ({skipped} skipped), {total_items} polygons")
    for name, count in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {name:28s} {count}")

    groups = list(names.values()) if names else ["lowgrade_dysplasia", "highgrade_dysplasia"]
    print(
        "\ntiling must be given exactly these names, or every tile is labeled negative:\n"
        f"  target_groups=[{','.join(groups)}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
