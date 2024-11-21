import logging
from itertools import product

import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F

from source.trainer import BaseTrainer
from source.nnet.utils.padding import lengths_to_padding_mask


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
        self.metrics_to_log = ["loss"]
        self.initialize_metrics()
        # Generate all combinations of azimuth and elevation and create label map
        elevation_grid, azimuth_grid = torch.meshgrid(model.elevation_angles, model.azimuth_angles, indexing="xy")
        self.angle_combinations = torch.stack([elevation_grid.flatten(), azimuth_grid.flatten()], dim=1)

    # Define custom model forward pass. Should always return: loss, predictions, targets
    # The forward pass will always receive a single element 'data' from the data loader.

    def forward_pass(self, data):
        # Unpack data batch
        input_data, lengths, targets = data

        # Compute targets
        matches = (self.angle_combinations[:, None, :] == targets[None, :, :]).all(
            dim=-1)  # Shape: (num_combinations, batch_size)
        matches = matches.to(torch.long)
        targets = matches.argmax(dim=0)

        # Transfer data to device
        input_data, lengths, targets = (input_data.to(self.device), lengths.to(self.device), targets.to(self.device))

        # Make prediction
        predictions, lengths = self.model(input_data, lengths=lengths)

        # Compute loss
        targets = targets.unsqueeze(1).repeat(1, predictions.size(1)) # Repeat target DoA for each time step
        padding_mask = lengths_to_padding_mask(lengths)
        targets[padding_mask] = -1
        loss = self.criterion(predictions.transpose(1, 2), targets)
        return loss, predictions, targets

    # Define custom function to compute metrics. Should always return a dict with 'metric_name: value' key/value pairs.
    def compute_batch_metrics(self, loss, predictions, targets):
        return {"loss": loss}

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

        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            batch_index = 0
            for batch_index, data in enumerate(self.val_loader):
                loss, _, _ = self.forward_pass(data)
                total_loss += loss.item()

        avg_loss = total_loss / (batch_index + 1)
        if self.use_wandb:
            wandb.log(data={"avg_val_loss": avg_loss},
                      step=self.steps)
        self.last_validation_step = self.steps
        logger.info(f"Average Validation Loss: {avg_loss:.3f}")
        return avg_loss


def setup_trainer(config, model, train_loader, val_loader, distributed):
    return DoATrainer(config, model, train_loader, val_loader, distributed)
