import logging
from itertools import product

import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import numpy as np

from source.trainer import BaseTrainer
from source.nnet.utils.masking import lengths_to_padding_mask
from source.metrics import angular_error, threshold_accuracy, angular_precision
from source.utils.spatial import spherical_to_cartesian


logger = logging.getLogger(__name__)


class DoATrainer(BaseTrainer):
    """
    Doa Trainer class for training DoA model.
    """

    def __init__(self,
                 config: DictConfig,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 distributed: bool):
        # Call the BaseTrainer initializer to inherit setup
        super().__init__(config, model, train_loader, val_loader, distributed)
        # Set experiment specific variable
        self.metrics_to_log = ["loss", "mean_angular_error", "std_angular_error", "accuracy@10degree", "median_angular_error"]
        self.initialize_metrics()
        # Generate all combinations of azimuth and elevation and create label map
        elevation_grid, azimuth_grid = torch.meshgrid(model.elevation_angles, model.azimuth_angles, indexing="xy")
        self.angle_combinations = torch.stack([elevation_grid.flatten(), azimuth_grid.flatten()], dim=1).to(self.device)

    # Define custom model forward pass. Should always return: loss, predictions, targets
    # The forward pass will always receive a single element 'data' from the data loader.

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

    # Define custom function to compute metrics. Should always return a dict with 'metric_name: value' key/value pairs.
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
                "median_angular_error": error.median(),}

    # Define custom validation loop
    def validate(self):
        """
        Validate the model on the validation dataset.
        :return: avg_loss: float
        """
        if self.val_loader is None:
            logger.info("Validation data not provided. Skipping validation.")
            return None
        logger.info("Evaluating model on validation set...")

        # Reset validation metrics
        self.validation_metrics = {metric: 0.0 for metric in self.validation_metrics.keys()}
        self.model.eval()
        with torch.no_grad():
            batch_index = 0
            for batch_index, data in enumerate(self.val_loader):
                loss, predictions, targets = self.forward_pass(data)
                batch_metrics = self.compute_batch_metrics(loss, predictions, targets)
                for metric, value in batch_metrics.items():
                    self.validation_metrics[metric] += value

        # Log average metrics
        avg_validation_metrics = {metric: value / (batch_index + 1) for metric, value in self.validation_metrics.items()}
        # Construct a log message with formatted metric values
        avg_val_metrics_message = ", ".join([f"{metric.lower()} = {value:.3f}"
                                             for metric, value in avg_validation_metrics.items()])
        logger.info("Average Validation Metrics - Epoch %s: %s", self.epoch, avg_val_metrics_message)
        # Log average metrics to wandb
        if self.use_wandb:
            avg_metrics_wandb = {"val_" + metric: value for metric, value in avg_validation_metrics.items()}
            wandb.log({**avg_metrics_wandb, "epoch": self.epoch}, step=self.steps)
        avg_val_score = avg_validation_metrics['loss']

        return avg_val_score


def setup_trainer(config, model, train_loader, val_loader, distributed):
    return DoATrainer(config, model, train_loader, val_loader, distributed)
