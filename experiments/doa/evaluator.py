import os
import time
from math import ceil, floor
import logging

import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from source.utils import seed_everything, get_device
from source.loss import get_loss

logger = logging.getLogger(__name__)


class ExampleEvaluator:
    """
    Example Trainer class for training encoder.
    """

    def __init__(self,
                 config: DictConfig,
                 model: torch.nn.Module,
                 test_loader: DataLoader):

        # Experiment settings
        self.seed = config.training.seed

