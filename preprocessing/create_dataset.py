import re
import pandas as pd
from pathlib import Path
import tempfile
import mlflow
import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def get_annot(folder_path: Path) -> pd.DataFrame:
    """Scans for .json files and stores their absolute paths."""
    data = []
    for json_path in folder_path.glob("*.json"):
        data.append(
            {"slide_id": json_path.stem, "annot_path": str(json_path.absolute())}
        )

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["slide_id", "annot_path"]).set_index("slide_id")
    return df.set_index("slide_id")


def get_slides(folder_path: Path, pattern: re.Pattern) -> pd.DataFrame:
    """Scans for .czi files and stores their absolute paths."""
    data = []
    for slide_path in folder_path.glob("*.czi"):
        if not pattern.search(slide_path.name):
            continue

        data.append(
            {"slide_id": slide_path.stem, "slide_path": str(slide_path.absolute())}
        )

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["slide_id", "slide_path"]).set_index("slide_id")
    return df.set_index("slide_id")


def create_dataset(
    slide_folder: str, annot_uri: str, pattern_str: str
) -> tuple[pd.DataFrame, list[str], list[str]]:

    slide_path = Path(slide_folder)
    annot_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=annot_uri))
    pattern = re.compile(pattern_str)

    slides_df = get_slides(slide_path, pattern)
    annot_df = get_annot(annot_path)

    dataset_df = slides_df.join(annot_df, how="outer")

    missing_slides = dataset_df[dataset_df["slide_path"].isna()].index.to_list()
    missing_labels = dataset_df[dataset_df["annot_path"].isna()].index.to_list()

    dataset_df = dataset_df.dropna(subset=["slide_path", "annot_path"])
    dataset_df = dataset_df[["slide_path"]]

    return dataset_df, missing_slides, missing_labels


@with_cli_args(["+preprocessing=create_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    dataset, missing_slides, missing_labels = create_dataset(
        config.data_path,
        config.annot_mlflow_uri,
        config.regex_pattern,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        output_path = tmpdir_path / "dataset.csv"
        dataset.to_csv(output_path, index=True)
        logger.log_artifact(str(output_path))

        def _log_missing_items(items: list[str], filename: str) -> None:
            if not items:
                return
            file_path = tmpdir_path / filename
            file_path.write_text("\n".join(items) + "\n")
            logger.log_artifact(str(file_path))

        _log_missing_items(missing_slides, "missing_slides.txt")
        _log_missing_items(missing_labels, "missing_labels.txt")


if __name__ == "__main__":
    main()
