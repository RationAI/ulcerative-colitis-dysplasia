from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.model_selection import train_test_split
from sklearn.model_selection import GroupKFold


def split_dataset(
    dataset: pd.DataFrame, splits: DictConfig, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train, test = train_test_split(
        dataset,
        train_size=splits["train"],
        random_state=random_state,
        groups=dataset["case_id"],
    )

    preliminary_size = splits["test_preliminary"] / (1.0 - splits["train"])
    test_preliminary, test_final = train_test_split(
        test,
        train_size=preliminary_size,
        random_state=random_state,
        groups=test["case_id"],
    )

    return train, test_preliminary, test_final


def add_folds(train: pd.DataFrame, n_folds: int) -> pd.DataFrame:

    splitter = GroupKFold(n_splits=n_folds)

    train["fold"] = -1
    for fold, (_, val_idx) in enumerate(splitter.split(train, groups=train["case_id"])):
        train.loc[train.iloc[val_idx].index, "fold"] = fold

    return train


@with_cli_args(["+preprocessing=split_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset = pd.read_csv(download_artifacts(config.mlflow_uris.dataset))

    train, test_preliminary, test_final = split_dataset(
        dataset, config.splits, config.random_state
    )

    train = add_folds(train, config.n_folds)

    with TemporaryDirectory() as tmpdir:
        for name, df in (
            ("train", train),
            ("test_preliminary", test_preliminary),
            ("test_final", test_final),
        ):
            output_path = Path(tmpdir) / f"{name}.csv"
            df.to_csv(output_path, index=False)
            logger.log_artifact(str(output_path))


if __name__ == "__main__":
    main()
