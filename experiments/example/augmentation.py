from logging import getLogger

from omegaconf import OmegaConf

from source.augment import get_composed_augmentations


logger = getLogger(__name__)


def get_augmentor(config):
    logger.info(f"Augmentor Config: {OmegaConf.to_object(config.augment.waveform)}")
    return get_composed_augmentations(OmegaConf.to_object(config.augment.waveform))
