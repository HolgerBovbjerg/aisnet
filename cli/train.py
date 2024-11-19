import logging

import hydra
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError

from hydra.core.hydra_config import HydraConfig

from cli.utils import load_experiment_run

# Setting up the logger in the main script
logger = logging.getLogger(__name__)

# Setup support for eval resolver for configs
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../experiments/example/configs", config_name="config", version_base=None)
def main(config: OmegaConf) -> None:
    """
    Main function for starting experiment run
    :param config: config for experiment
    :return: None
    """

    run = None
    try:
        experiment_type = config.experiment_type
        run = load_experiment_run(experiment_type)
    except ValueError as e:
        print(e)

    # Create trainer and start training
    try:
        distributed = config.job.device.name == "distributed"
    except ConfigAttributeError as e:
        print("Did not find config.job.device.name in config, defaulting to non-distributed training.")
        print(e)
        distributed = False

    if distributed:
        if HydraConfig.initialized():
            with open_dict(config):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                config.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )
        logger.info("Using distributed training.")
        run(config, distributed=distributed)
    else:
        logger.info("Using single node training.")
        run(config, distributed=distributed)


if __name__ == "__main__":
    main()
