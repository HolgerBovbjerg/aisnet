import os
from functools import partial
import logging
from typing import Union

import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import pandas as pd
import numpy as np

from source.utils import seed_everything, get_device
from source.loss import get_loss
from .model import build_model
from .data_loading import generate_data_set, get_data_loader, pad_collate
from source.nnet.utils.masking import lengths_to_padding_mask
from source.metrics import angular_error, threshold_accuracy
from source.utils.spatial import spherical_to_cartesian
from source.augment.add_noise import AddNoise

logger = logging.getLogger(__name__)


def identity(x):
    return x


class DoAEvaluator:
    """
    Doa Evaluator class for evaluating DoA prediction model.
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
        self.seed = config.evaluation.seed
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

        self.evaluation_dir = os.path.join(str(self.save_dir), "test_scores")
        os.makedirs(self.evaluation_dir, exist_ok=True)

        # Evaluation metrics
        # Set up the loss
        loss_config = OmegaConf.to_object(config.loss)
        logger.info(f"Loss config: {loss_config}")
        self.criterion = get_loss(**loss_config)
        # Setup metrics
        self.metrics_to_log = ["loss", "mean_angular_error", "std_angular_error", "accuracy@10degree", "median_angular_error"]
        self.metrics_df = pd.DataFrame()

        # Evaluation settings
        self.noise_types = OmegaConf.to_object(config.evaluation.noise)
        self.snr_range = config.evaluation.snrs

        # Generate all combinations of azimuth and elevation and create label map
        elevation_grid, azimuth_grid = torch.meshgrid(self.model.elevation_angles, self.model.azimuth_angles, indexing="xy")
        self.angle_combinations = torch.stack([elevation_grid.flatten(), azimuth_grid.flatten()], dim=1).to(self.device)

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
                       name=config.wandb.name + "_test",
                       group=config.wandb.group,
                       job_type=config.wandb.job_type,
                       tags=config.wandb.tags + ["test"] if config.wandb.tags else ["test"])
            # resume=config.wandb.resume)
            # id=self.wandb_id)
            wandb.config.update(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))

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

    def _build_test_loader(self, config, augmentor):
        logger.info("Creating test dataset...")
        test_config = config.data.test
        test_dataset = generate_data_set(test_config)
        if self.distributed:
            # Distributed Sampler: Ensures data is divided among GPUs using DistributedSampler.
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["SLURM_PROCID"])
            test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)
        else:
            test_sampler = None
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"] if self.distributed else config.job.num_workers)
        test_loader = get_data_loader(test_dataset, batch_size=config.data.test.batch_size,
                                      shuffle=False, sampler=test_sampler,
                                      pin_memory=config.job.pin_memory,
                                      collate_fn=partial(pad_collate, augmentor=augmentor),
                                      num_workers=num_workers)
        return test_loader

    def forward_pass(self, data):
        # Unpack data batch
        input_data, lengths, targets = data

        # Transfer data to device
        input_data, lengths, targets = (input_data.to(self.device), lengths.to(self.device), targets.to(self.device))

        # Compute targets
        matches = (self.angle_combinations[:, None, :] == targets[None, :, :]).all(
            dim=-1)  # Shape: (num_combinations, batch_size)
        matches = matches.to(torch.long)
        targets = matches.argmax(dim=0)

        # Make prediction
        predictions, lengths = self.model(input_data, lengths=lengths)

        # Compute loss
        targets = targets.unsqueeze(1).repeat(1, predictions.size(1)) # Repeat target DoA for each time step (assuming static source)
        padding_mask = lengths_to_padding_mask(lengths)
        predictions = predictions[~padding_mask]
        targets = targets[~padding_mask]
        loss = self.criterion(predictions, targets)
        return loss, predictions, targets

    def compute_batch_metrics(self, loss, predictions, targets):
        predicted_angle = predictions.argmax(dim=-1)
        predicted_angle = predicted_angle[targets != -1]
        predicted_angle = self.angle_combinations[predicted_angle]
        targets = targets[targets != -1]
        targets = self.angle_combinations[targets]

        predicted_angle = spherical_to_cartesian(torch.ones(predicted_angle.size(0), device=predicted_angle.device),
                                                 torch.deg2rad(predicted_angle[:, 0]),
                                                 torch.deg2rad(predicted_angle[:, 1]))
        targets = spherical_to_cartesian(torch.ones(targets.size(0), device=targets.device),
                                         torch.deg2rad(targets[:, 0]),
                                         torch.deg2rad(targets[:, 1]))

        error = torch.rad2deg(angular_error(predicted_angle, targets))
        accuracy = threshold_accuracy(error, threshold=np.deg2rad(10.))

        return {"loss": loss,
                "mean_angular_error": error.mean(),
                "accuracy@10degree": accuracy,
                "std_angular_error": error.std(),
                "median_angular_error": error.median(), }

    @torch.no_grad()
    def _evaluate(self, test_loader, noise_type: str = "", snr: Union[float, int] = np.inf):
        # Initialize metrics
        accumulated_metrics = {metric: 0.0 for metric in self.metrics_to_log}
        accumulated_steps = 0
        for data in test_loader:
            loss, predictions, targets = self.forward_pass(data)
            batch_metrics = self.compute_batch_metrics(loss, predictions, targets)
            for metric, value in batch_metrics.items():
                accumulated_metrics[metric] += value.item()
            accumulated_steps += 1

        # Compute average metrics
        avg_metrics = {}
        for metric, value in accumulated_metrics.items():
            avg_metrics[metric] = value / accumulated_steps

        # Save and log metrics
        avg_test_metrics_message = ", ".join([f"{metric.lower()} = {value:.3f}"
                                              for metric, value in avg_metrics.items()])
        logger.info("Average Test Metrics: %s", avg_test_metrics_message)

        avg_metrics = {metric: [value] for metric, value in avg_metrics.items()}
        avg_metrics["snr_db"] = snr
        avg_metrics["noise_type"] = noise_type

        # Update metrics
        metrics_df = pd.DataFrame.from_dict(avg_metrics)
        self.metrics_df = pd.concat([self.metrics_df, metrics_df], ignore_index=True)

        # Log to W&B dynamically
        if self.use_wandb:
            metrics_table = wandb.Table(dataframe=self.metrics_df)
            wandb.log({"test_metrics": metrics_table})

        # Save the DataFrame to a CSV dynamically
        logger.info("Saving test results in %s", os.path.join(self.evaluation_dir, "test_metrics.csv"))
        self.metrics_df.to_csv(os.path.join(self.evaluation_dir, "test_metrics.csv"), index=False)


    def evaluate(self):
        logger.info("Starting model evaluation:")
        logger.info(f"Data Config: {OmegaConf.to_object(self.config.data)}")
        n_evaluation_settings = len(list(self.noise_types.keys())) + 1
        noise_types = list(self.noise_types.keys())
        logger.info(f"Evaluating model on clean data and noise types: {noise_types} at SNRs: {self.snr_range}")

        for i in range(n_evaluation_settings):
            if i == 0:
                seed_everything(self.seed)
                logger.info("Evaluating on clean data")
                test_loader = self._build_test_loader(self.config, augmentor=identity)
                self._evaluate(test_loader, noise_type="clean", snr=np.inf)
            else:
                for snr in self.snr_range:
                    seed_everything(self.seed)
                    path = self.noise_types[noise_types[i]]["path"]
                    logger.info("Evaluating noise type: %s at SNR = %s dB", noise_types[i-1], snr)
                    augment_config = {"noise_paths": [path], "sampling_rate": 16000, "snr_db_min": snr, "snr_db_max": snr, "p": 1.}
                    augmentor = AddNoise(**augment_config)
                    test_loader = self._build_test_loader(self.config, augmentor=augmentor)
                    self._evaluate(test_loader, noise_type=noise_types[i-1], snr=snr)
        print("Evaluation completed.")


def setup_evaluator(config, distributed: bool = False):
    return DoAEvaluator(config, distributed=distributed)
