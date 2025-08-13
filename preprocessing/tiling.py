import hydra
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.mlkit import autolog


@hydra.main(config_path="./configs", config_name="preproessing", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    logger.experiment.log_artifacts(
        logger.run_id, config.tiling.src_path, config.tiling.dest_path
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
