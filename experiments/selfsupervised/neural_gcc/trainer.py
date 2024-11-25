import logging

import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn import functional as F

from source.metrics import equal_error_rate, accuracy
from source.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class NeuralGCCTrainer(BaseTrainer):
    """
    Example Trainer class for training example model.
    """

    def __init__(self,
                 config: DictConfig,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 distributed: bool = False):
        # Call the BaseTrainer initializer to inherit setup
        super().__init__(config, model, train_loader, val_loader, distributed)
        # Set experiment specific variable
        self.metrics_to_log = ["loss"]
        self.initialize_metrics()

    # Define custom model forward pass. Should always return: loss, predictions, targets
    # The forward pass will always receive a single element 'data' from the data loader.
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
        loss = self.criterion(predictions, targets)
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

        avg_eer = total_loss / (batch_index + 1)
        if self.use_wandb:
            wandb.log(data={"avg_val_loss": avg_eer},
                      step=self.steps)
        self.last_validation_step = self.steps
        logger.info(f"Average Validation Loss: {avg_eer:.3f}")
        return avg_eer


def setup_trainer(config, model, train_loader, val_loader, distributed):
    return NeuralGCCTrainer(config, model, train_loader, val_loader, distributed)
