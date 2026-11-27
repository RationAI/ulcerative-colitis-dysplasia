"""Shared helpers for cohort dataset readers."""

import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def get_df(
    folder_path: Path,
    pattern: re.Pattern[str],
    key: str,
    ext: str,
    extract_case_id: Callable[[str], str],
) -> pd.DataFrame:
    """One row per file under folder_path matching pattern, indexed by slide_id.

    Files not matching the cohort regex are reported and skipped rather than
    failing the run, so a stray file cannot break dataset creation.
    """
    data = []
    skipped = []
    for slide_path in sorted(folder_path.glob(ext)):
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
