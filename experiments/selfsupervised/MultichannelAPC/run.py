from logging import getLogger

from .data_preparation import prepare_data
from .data_loading import setup_dataloader
from .model import build_model
from .trainer import setup_trainer

logger = getLogger(__name__)


def run(config):
    start_stage = config.start_stage
    stop_stage = config.stop_stage
    stage = start_stage

    while stage <= stop_stage:
        if stage == 0:
            logger.info("Stage {}: Data preparation:".format(stage))
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

            logger.info("All stages finished.")
            break

    logger.info("Experiment done.")