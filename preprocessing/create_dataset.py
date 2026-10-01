import re
import tempfile
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

#: Per-tile label sets live one dir per case under each zoom part:
#: ML_dataset/<zoom_part>/<NNN>/<NNN>_labels.csv
TILES_GLOB = "*/[0-9][0-9][0-9]/[0-9][0-9][0-9]_labels.csv"


def extract_case_id(stem: str) -> str:
    #: HunCRC has one slide per patient, so case_id == slide_id. The column is kept
    #: rather than dropped so split_dataset.py's group-by-case_id still works.
    return stem


def get_df(
    folder_path: Path, pattern: re.Pattern[str], key: str, ext: str
) -> pd.DataFrame:
    data = []
    skipped = []
    for slide_path in folder_path.glob(ext):
        if not pattern.search(slide_path.name):
            skipped.append(slide_path.name)
            continue

        data.append(
            {
                "slide_id": slide_path.stem,
                "case_id": extract_case_id(slide_path.stem),
                f"{key}_path": str(slide_path.absolute()),
            }
        )

    if skipped:
        print(
            f"[create_dataset] {key}: {len(skipped)} {ext} entries under "
            f"{folder_path} did not match {pattern.pattern} and were skipped: "
            f"{', '.join(sorted(skipped)[:10])}"
        )

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["slide_id", "case_id", f"{key}_path"]).set_index(
            "slide_id"
        )
    return df.set_index("slide_id")


def get_tile_df(tiles_path: Path) -> pd.DataFrame:
    """One row per slide, keyed by slide_id, pointing at its <NNN>_labels.csv.

    No regex filter here, unlike the slides: TILES_GLOB already pins the exact
    3-digit case dir and 3-digit label filename, and it cannot share a pattern with
    the *.mrxs names.

    Cases are split across zoom_1_partA/B/C + zoom_2, so every part is searched. A
    slide can have a label set in both a zoom_1 part and zoom_2, so zoom_1 (level 1)
    wins and zoom_2 only fills the gaps.
    """
    best: dict[str, tuple[int, str]] = {}
    for labels_path in sorted(tiles_path.glob(TILES_GLOB)):
        slide_id = labels_path.stem.removesuffix("_labels")
        #: .../<zoom_part>/<NNN>/<NNN>_labels.csv — the part is the grandparent.
        part = labels_path.parent.parent.name
        priority = 0 if part.startswith("zoom_1") else 1
        if slide_id not in best or priority < best[slide_id][0]:
            best[slide_id] = (priority, str(labels_path.absolute()))

    df = pd.DataFrame(
        [
            {"slide_id": slide_id, "tiles_path": path}
            for slide_id, (_, path) in best.items()
        ]
    )
    if df.empty:
        return pd.DataFrame(columns=["slide_id", "tiles_path"]).set_index("slide_id")
    return df.set_index("slide_id")


def create_dataset(
    slides_path: str,
    tiles_path: str,
    pattern_str: str,
) -> tuple[pd.DataFrame, list[str]]:
    pattern = re.compile(pattern_str)

    slides_df = get_df(Path(slides_path), pattern, key="slide", ext="*.mrxs")
    tiles_df = get_tile_df(Path(tiles_path))

    #: Left join keyed on slides, so a slide without a label set survives as NaN and
    #: becomes the missing_tiles QC signal instead of silently vanishing from the CSV.
    dataset_df = slides_df.join(tiles_df, how="left")

    missing_tiles = dataset_df[dataset_df["tiles_path"].isna()].index.to_list()
    print(
        f"[create_dataset] {len(dataset_df)} slides, "
        f"{len(missing_tiles)} without per-tile labels"
    )

    dataset_df = dataset_df[["case_id", "slide_path", "tiles_path"]]

    return dataset_df, missing_tiles


@with_cli_args(["+preprocessing=create_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset, missing_tiles = create_dataset(
        config.dataset.slides_path,
        config.dataset.tiles_path,
        config.dataset.regex_pattern,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        output_path = tmpdir_path / "dataset.csv"
        dataset.to_csv(output_path, index=True)
        logger.log_artifact(str(output_path))

        def _log_missing_items(items: list[str], filename: str) -> None:
            file_path = tmpdir_path / filename
            file_path.write_text("\n".join(items) + "\n")
            logger.log_artifact(str(file_path))

        _log_missing_items(missing_tiles, "missing_tiles.txt")


if __name__ == "__main__":
    main()
