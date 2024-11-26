import os
import time
from math import ceil, floor
import logging

import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import pandas as pd

from source.utils import seed_everything, get_device
from source.loss import get_loss
from .model import build_model
from .augmentation import get_augmentor
from .data_loading import create_dataset, get_data_loader, pad_collate_clean_noisy


logger = logging.getLogger(__name__)


class NeuralGCCEvaluator:
    """
    Example Trainer class for training encoder.
    """

    def __init__(self,
                 config: DictConfig,
                 distributed: bool = False):
        # Experiment settings
        self.distributed = distributed
        if self.distributed:
            self.rank = int(os.environ["SLURM_PROCID"])
            gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
            self.local_rank = self.rank - gpus_per_node * (self.rank // gpus_per_node)
        else:
            self.rank = 0
        self.seed = config.training.seed
        seed_everything(self.seed)
        self.device = self.local_rank if self.distributed else get_device(config.job.device.name)
        if torch.cuda.is_available() and distributed:
            torch.cuda.set_device(self.device)
        self.save_dir = os.path.join(config.exp_dir, config.exp_name)
        if not os.path.exists(self.save_dir):
            logger.error("Make sure directory %s exists, and contains a model to evaluate.", self.save_dir)
            raise ValueError("Model save directory %s does not exist. Cannot evaluate model.", self.save_dir)

        logger.info('Saving results in %s.', self.save_dir)
        self.config = config
        self.model = self._load_model()
        self.model.to(self.device)
        if distributed:
            # Distributes model across GPUs using DistributedDataParallel.
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])

        # Evaluation metrics
        # Set up the loss
        loss_config = OmegaConf.to_object(config.loss)
        logger.info(f"Loss config: {loss_config}")
        self.criterion = get_loss(**loss_config)
        # Setup metrics
        self.metrics_to_log = ["loss", "prediction_var", "target_var"]

        # Wandb settings
        self.use_wandb = config.wandb.enabled
        if self.use_wandb:
            # if self.wandb_id is None:
            # self.wandb_id = wandb.util.generate_id()
            self.setup_wandb(config)

    def setup_wandb(self, config):
        if self.rank == 0:
            wandb.init(project=config.wandb.project,
                       entity=config.wandb.entity,
                       name=config.wandb.name,
                       group=config.wandb.group,
                       job_type=config.wandb.job_type,
                       tags=config.wandb.tags)
                       #resume=config.wandb.resume)
                       #id=self.wandb_id)
            wandb.config.update(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
            if config.wandb.watch_model:
                wandb.watch(models=self.model,
                            log=config.wandb.watch_model_log,
                            log_freq=config.wandb.watch_model_log_frequency)

    def _load_model(self):
        print("Loading model from checkpoint...")
        model = build_model(self.config)
        checkpoint_path = os.path.join(self.save_dir, "best_model.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if self.distributed:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        return model

    def _build_test_loader(self, config):
        logger.info("Creating test data augmentor.")
        augmentor = get_augmentor(config)
        logger.info("Creating test datasets...")
        logger.info(f"Data Config: {OmegaConf.to_object(config.data)}")
        test_config = config.data.train
        test_dataset= create_dataset(test_config, augmentor)
        if self.distributed:
            # Distributed Sampler: Ensures data is divided among GPUs using DistributedSampler.
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["SLURM_PROCID"])
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)
        else:
            test_sampler = None
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"] if self.distributed else config.job.num_workers)
        test_loader = get_data_loader(test_dataset, batch_size=config.training.batch_size,
                                      shuffle=False, sampler=test_sampler,
                                      pin_memory=config.job.pin_memory,
                                      collate_fn=pad_collate_clean_noisy,
                                      num_workers=num_workers)
        return test_loader

    def forward_pass(self, data):
        # Unpack data batch
        if len(data) == 2:
            input_data, lengths = data
            targets = None
        else:
            input_data, lengths, targets = data

        # Transfer data to device
        input_data, lengths = (input_data.to(self.device), lengths.to(self.device))
        if targets is not None:
            targets = targets.to(self.device)
        else:
            targets = input_data.detach().clone()

        # Make prediction
        predictions, targets, lengths = self.model(input_data, lengths=lengths, target=targets)

        # Compute loss
        loss = self.criterion(predictions, targets) / lengths.sum()
        return loss, predictions, targets

    # Define custom function to compute metrics. Should always return a dict with 'metric_name: value' key/value pairs.
    def compute_batch_metrics(self, loss, predictions, targets):
        prediction_var = predictions.var()
        target_var = targets.var()
        return {"loss": loss, "prediction_var": prediction_var, "target_var": target_var}

    @torch.no_grad()
    def evaluate(self):
        print("Evaluating the model...")
        # Should support multiple data loaders, e.g., in the case of evaluation with variable SNR
        test_loader = self._build_test_loader(self.config)
        # Initialize metrics
        accumulated_metrics = {metric: 0.0 for metric in self.metrics_to_log}
        accumulated_steps = 0
        for data in test_loader:
            loss, predictions, targets = self.forward_pass(data)
            batch_metrics = self.compute_batch_metrics(loss, predictions, targets)
            for metric, value in batch_metrics.items():
                accumulated_metrics[metric] += value.item()
                accumulated_steps += 1

        avg_metrics = {}
        for metric, value in accumulated_metrics.items():
            avg_metrics[metric] = [value / accumulated_steps]

        avg_test_metrics_message = ", ".join([f"{metric.lower()} = {value:.3f}"
                                             for metric, value in avg_metrics.items()])
        logger.info("Average Test Metrics: %s", avg_test_metrics_message)

        # Save and log metrics
        metrics_df = pd.DataFrame.from_dict(avg_metrics)
        metrics_df.to_csv(os.path.join(str(self.save_dir), "test_metrics.csv"), index=False)

        print("Evaluation completed.")


def setup_evaluator(config, distributed: bool = False):
    return NeuralGCCEvaluator(config, distributed=distributed)
