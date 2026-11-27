"""Create dataset.csv for the cohort given by config.dataset.cohort.

The cohort specific logic (case model, slide discovery, annotation source)
lives in preprocessing/readers/<cohort>.py; this stage only dispatches and
logs the shared contract: dataset.csv (indexed by slide_id) plus a missing
items file as the QC signal.
"""

import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

try:
    # module mode: uv run python -m preprocessing.create_dataset (submitter scripts)
    from preprocessing.readers import get_reader
except ImportError:
    # script mode: uv run python preprocessing/create_dataset.py
    from readers import get_reader


@with_cli_args(["+preprocessing=create_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    reader = get_reader(config.dataset.cohort)
    dataset, missing = reader.create_dataset(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        output_path = tmpdir_path / "dataset.csv"
        dataset.to_csv(output_path, index=True)
        logger.log_artifact(str(output_path))

        missing_path = tmpdir_path / "missing.txt"
        missing_path.write_text("\n".join(missing) + "\n")
        logger.log_artifact(str(missing_path))


if __name__ == "__main__":
    main()
