import logging

import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn import functional as F

from source.metrics import equal_error_rate, accuracy
from source.trainer import BaseTrainer

logger = logging.getLogger(__name__)


def transform_speaker_ids_to_labels(speaker_ids_tensor, label_mapping):
    """Transforms a tensor of speaker IDs into corresponding class labels."""
    # Create a mapping tensor based on the provided dictionary
    # Assuming label_mapping is in the format {speaker_id: class_label}
    max_speaker_id = max(label_mapping.keys())

    # Create a tensor to hold the class labels, initializing with -1 (or any invalid label)
    label_tensor = torch.full((max_speaker_id + 1,), -1, dtype=torch.long, device=speaker_ids_tensor.device)

    # Populate the tensor with the class labels based on the label mapping
    for speaker_id, class_label in label_mapping.items():
        label_tensor[speaker_id] = class_label

    # Use the label tensor to transform the input speaker_ids_tensor
    transformed_labels = label_tensor[speaker_ids_tensor]

    return transformed_labels


class ExampleTrainer(BaseTrainer):
    """
    Example Trainer class for training example model.
    """

    def __init__(self,
                 config: DictConfig,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 label_mapping: dict,
                 distributed: bool = False):
        # Call the BaseTrainer initializer to inherit setup
        super().__init__(config, model, train_loader, val_loader, distributed)
        # Set experiment specific variable
        self.label_mapping = label_mapping
        self.metrics_to_log = ["loss", "accuracy", "EER"]
        self.initialize_metrics()

    # Define custom model forward pass. Should always return: loss, predictions, targets
    # The forward pass will always receive a single element 'data' from the data loader.
    def forward_pass(self, data):
        # Unpack data batch
        input_data, lengths, targets = data
        # Transfer data to device
        input_data, lengths, targets = (input_data.to(self.device),
                                        lengths.to(self.device),
                                        targets.to(self.device))
        # Make prediction
        predictions = self.model(input_data, lengths=lengths)
        # Generate target labels
        targets = transform_speaker_ids_to_labels(targets, self.label_mapping)
        # Compute loss
        loss = self.criterion(predictions, targets)
        return loss, predictions, targets

    # Define custom function to compute metrics. Should always return a dict with 'metric_name: value' key/value pairs.
    def compute_batch_metrics(self, loss, predictions, targets):
        predictions, targets = predictions.to("cpu"), targets.to("cpu")
        return {"loss": loss,
                "accuracy": accuracy(predictions, targets),
                "EER": equal_error_rate(predictions.numpy(), F.one_hot(targets, num_classes=predictions.size(-1)).numpy())}

    # Define custom validation loop
    def validate(self):
        """
        Validate the model on the validation dataset.
        :return: avg_loss: float
        """
        if self.val_loader is None:
            logger.info("Validation data not provided. Skipping validation.")
            return None
        else:
            logger.info("Evaluating model on validation set...")

        self.model.eval()
        with ((torch.no_grad())):
            total_eer = 0.0
            batch_index = 0
            for batch_index, data in enumerate(self.val_loader):
                ref_waveforms_padded, ref_lengths = data[0], data[1]
                pos_waveforms_padded, pos_lengths = data[2], data[3]
                neg_waveforms_padded, neg_lengths = data[4], data[5]

                # Forward pass
                ref_waveforms_padded, ref_lengths = ref_waveforms_padded.to(self.device), ref_lengths.to(self.device)
                ref_embedding = self.model.embed(ref_waveforms_padded, ref_lengths)

                pos_waveforms_padded, pos_lengths = pos_waveforms_padded.to(self.device), pos_lengths.to(self.device)
                pos_embedding = self.model.embed(pos_waveforms_padded, pos_lengths)

                neg_waveforms_padded, neg_lengths = neg_waveforms_padded.to(self.device), neg_lengths.to(self.device)
                neg_embedding = self.model.embed(neg_waveforms_padded, neg_lengths)

                # Compute loss
                pos_similarity = torch.nn.functional.cosine_similarity(ref_embedding, pos_embedding)
                neg_similarity = torch.nn.functional.cosine_similarity(ref_embedding, neg_embedding)

                pos_labels = torch.ones_like(pos_similarity)
                neg_labels = torch.zeros_like(neg_similarity)

                predictions = torch.cat([pos_similarity, neg_similarity])
                targets = torch.cat([pos_labels, neg_labels])

                eer = equal_error_rate(predictions.cpu().numpy(), targets.cpu().numpy())

                # Accumulate
                total_eer += eer

        avg_eer = total_eer / (batch_index + 1)
        if self.use_wandb:
            wandb.log(data={"avg_val_eer": avg_eer},
                      step=self.steps)
        self.last_validation_step = self.steps
        logger.info(f"Average Validation EER: {avg_eer:.3f}")
        return avg_eer


def setup_trainer(config, model, train_loader, val_loader, label_mapping, distributed):
    return ExampleTrainer(config, model, train_loader, val_loader, label_mapping, distributed)
