"""IKEM cohort reader.

Cases come from two Excel selection sheets (clear / unclear); slides are CZIs
named <patient>_<section>_HE_... and annotations are the EMPIA JSONs next to
them. A selected case without an annotation JSON is a NEGATIVE slide.
"""

import re
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from .common import get_df


def extract_case_id(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid slide/annotation name format: {stem}")
    return f"{parts[0]}/{parts[1]}"


def load_selected_cases(excel_path: str, label: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, header=None)
    cases = df.iloc[:, 1].dropna().astype(str).str.strip()
    cases = cases[cases != "Bs"].str.replace("_", "/", regex=False).tolist()
    return pd.DataFrame({"case_id": cases, "clarity": label})


def create_dataset(config: DictConfig) -> tuple[pd.DataFrame, list[str]]:
    """Build the dataset for the IKEM cohort.

    Returns (dataset_df indexed by slide_id with [case_id, slide_path,
    annot_path, clarity], missing slides) -- same contract as the other
    readers.
    """
    pattern = re.compile(config.dataset.regex_pattern)

    slides_df = get_df(
        Path(config.dataset.data_path), pattern, key="slide", ext="*.czi",
        extract_case_id=extract_case_id,
    )
    annot_df = get_df(
        Path(config.dataset.annot_path), pattern, key="annot", ext="*.json",
        extract_case_id=extract_case_id,
    )

    clear_cases = load_selected_cases(config.dataset.selected_clear_cases, "clear")
    unclear_cases = load_selected_cases(config.dataset.selected_unclear_cases, "unclear")
    selected_cases = pd.concat([clear_cases, unclear_cases], ignore_index=True)

    dataset_df = slides_df.join(annot_df, how="outer", rsuffix="_drop")
    dataset_df["case_id"] = dataset_df["case_id"].fillna(dataset_df["case_id_drop"])

    dataset_df = dataset_df.reset_index().merge(
        selected_cases, on="case_id", how="right"
    )

    missing_slides = (
        dataset_df[dataset_df["slide_path"].isna()]["case_id"].dropna().to_list()
    )

    dataset_df = dataset_df.dropna(subset=["slide_path"])

    dataset_df["annot_path"] = dataset_df["annot_path"].fillna("NEGATIVE")

    dataset_df = dataset_df.set_index("slide_id")
    dataset_df = dataset_df[["case_id", "slide_path", "annot_path", "clarity"]]

    return dataset_df, missing_slides
