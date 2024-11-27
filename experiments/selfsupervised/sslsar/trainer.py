import logging

import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from source.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SSLSARTrainer(BaseTrainer):
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
        self.metrics_to_log = ["loss", "prediction_var", "target_var"]
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
        prediction_var = predictions.var()
        target_var = targets.var()
        return {"loss": loss, "prediction_var": prediction_var, "target_var": target_var}

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
                    self.validation_metrics[metric] += value.item()

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
    return SSLSARTrainer(config, model, train_loader, val_loader, distributed)
