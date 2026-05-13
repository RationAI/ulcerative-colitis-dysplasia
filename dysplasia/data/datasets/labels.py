from enum import Enum
from pathlib import Path

import pandas as pd


def process_slides(slides: pd.DataFrame) -> pd.DataFrame:
    # `rationai.mlkit` loads slides as a HuggingFace `datasets.Dataset`.
    # Convert to pandas for downstream code that expects `.apply()` / `.iterrows()`.
    if not isinstance(slides, pd.DataFrame):
        if hasattr(slides, "to_pandas"):
            slides = slides.to_pandas()
        else:
            slides = pd.DataFrame(slides)

    slides["name"] = slides["path"].apply(lambda x: Path(x).stem)
    return slides
