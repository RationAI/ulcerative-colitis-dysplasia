import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@hydra.main(
    config_path="./configs", config_name="preproessing/tiling", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # TODO: Implement tiling logic using rationai.tiling library
    # Reference: https://rationai.gitlab-pages.ics.muni.cz/digital-pathology/libraries/tiling/
    logger.log_artifacts(config.data_path, config.artifact_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
