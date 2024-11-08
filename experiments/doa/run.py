from logging import getLogger

from experiments.doa.data_preparation import prepare_data
from experiments.doa.data_loading import setup_dataloader
from experiments.doa.augmentation import get_augmentor
from experiments.doa.model import build_model
from experiments.doa.trainer import setup_trainer

logger = getLogger(__name__)


def run(config):
    start_stage = config.start_stage
    stop_stage = config.stop_stage
    stage = start_stage
    logger.info(f"Starting example 2")
    while stage <= stop_stage:
        if stage == 0:
            logger.info("Stage {}: Data preparation".format(stage))
            prepare_data(config)
            stage += 1
        if stage == 1:
            logger.info("Stage {}: Model training".format(stage))
            logger.info("Setting up data loaders.")
            train_loader, val_loader = setup_dataloader(config)
            logger.info("Building model.")
            model = build_model(config)
            logger.info("Setting up trainer.")
            trainer = setup_trainer(config, model, train_loader, val_loader)
            logger.info("Starting training.")
            trainer.train()
            logger.info("Finished training.")
            stage += 1
        if stage == 2:
            logger.info("Stage {}: Evaluating trained model".format(stage))

            logger.info("Finished last stage: {}".format(stage))
            break

    logger.info("Finished experiment")
