import importlib
from logging import getLogger

logger = getLogger(__name__)


def load_experiment_run(experiment_type):
    try:
        logger.info(f"Trying to load run module 'experiments.{experiment_type}.run'")
        experiment_module = importlib.import_module(f"experiments.{experiment_type}.run")
        logger.info(f"Loaded run module from 'experiments.{experiment_type}.run'")
        return experiment_module.run
    except ModuleNotFoundError as e:
        error_message = f"Module import error. Error: {str(e)}"
        logger.error(error_message)
        raise ModuleNotFoundError(error_message)


