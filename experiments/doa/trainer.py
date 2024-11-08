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
from source.optimizer import get_optimizer
from source.scheduler import get_scheduler

logger = logging.getLogger(__name__)


class ExampleTrainer:
    """
    Example Trainer class for training encoder.
    """

    def __init__(self,
                 config: DictConfig,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader):

        # Experiment settings
        self.seed = config.training.seed
        seed_everything(self.seed)
        self.device = get_device(config.job.device)
        self.save_dir = os.path.join(config.job.exp_dir, config.job.exp_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logger.info(f'Saving results in {self.save_dir}')

        # Training settings
        self.num_epochs = config.training.num_epochs
        self.clip_grad_norm = config.training.clip_grad_norm
        if self.clip_grad_norm:
            logger.info(f'Clipping gradients at {self.clip_grad_norm}.')
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        if self.gradient_accumulation_steps > 1:
            logger.info(f'Using {self.gradient_accumulation_steps} gradient accumulation steps.')

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Send model to device
        self.model = model
        self.elevation_angles = model.elevation_angles
        self.azimuth_angles = model.azimuth_angles
        self.model.to(self.device)

        # Set up the optimizer
        logger.info(f"Optimizer config: {OmegaConf.to_object(config.optimizer)}")
        self.optimizer = get_optimizer(self.model, opt_config=config.optimizer)

        # Set up the loss function
        logger.info(f"Loss config: {OmegaConf.to_object(config.loss)}")
        self.criterion = get_loss(**config.loss)

        # Setup lr scheduler
        if config.scheduler is not None:
            logger.info(f"Scheduler config: {OmegaConf.to_object(config.scheduler)}")
            if config.scheduler.scheduler_type is not None:
                if config.scheduler.steps_per_epoch is not None:
                    total_iters = config.scheduler.steps_per_epoch * max(1, config.training.num_epochs)
                    self.scheduler = get_scheduler(self.optimizer,
                                                   scheduler_type=config.scheduler.scheduler_type,
                                                   t_max=total_iters,
                                                   **config.scheduler.scheduler_kwargs)
                else:
                    steps_per_epoch = ceil(len(self.train_loader) / config.loss.accumulation_steps)
                    total_iters = steps_per_epoch * max(1, config.training.num_epochs)

                    self.scheduler = get_scheduler(self.optimizer,
                                                   scheduler_type=config.scheduler.scheduler_type,
                                                   t_max=total_iters,
                                                   **config.scheduler.scheduler_kwargs)
            else:
                self.scheduler = None
        else:
            self.scheduler = None

        # Job settings
        self.log_interval = config.job.log_interval  # Log every N batches
        self.validation_interval = config.job.validation_interval  # Validate every N batches
        # only used if smaller than number of batches. Ignored if None.

        # Wandb settings
        self.use_wandb = config.wandb.enabled
        if self.use_wandb:
            wandb.init(project=config.wandb.project,
                       entity=config.wandb.entity,
                       name=config.wandb.name,
                       group=config.wandb.group,
                       job_type=config.wandb.job_type,
                       tags=config.wandb.tags)
            wandb.config.update(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))

        # Initialize training variables
        self.start_epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')

        # Check for existing checkpoint
        self.checkpoint_path = os.path.join(self.save_dir, "checkpoint.pth")
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def _create_targets(self, targets):
        batch_size, time_steps, _ = targets.shape

        # Find indices for each target elevation and azimuth
        elevation_indices = (targets == self.elevation_angles)
        azimuth_indices = (targets == self.azimuth_angles)

        # Create the one-hot tensor
        one_hot_vectors = torch.zeros(batch_size, time_steps, len(self.elevation_angles), len(self.azimuth_angles))
        one_hot_vectors[elevation_indices] = 1
        one_hot_vectors = one_hot_vectors.transpose(-1, -2)
        one_hot_vectors[azimuth_indices] = 1
        one_hot_vectors = one_hot_vectors.transpose(-1, -2)

        return one_hot_vectors
    def train_one_batch(self, batch_idx, data):
        """
        Trains the model for one batch.
        :param batch_idx: int
        :param data: tuple of data from data loader
        :return: (loss: float, target_var: float, prediction_var: float)
        """
        input_data, lengths, targets = data
        input_data, lengths, targets = (input_data.to(self.device),
                                        lengths.to(self.device),
                                        targets.to(self.device))

        self.optimizer.zero_grad()

        # Forward pass
        predictions = self.model(input_data, lengths=lengths)

        # Compute loss and gradients
        targets = targets.unsqueeze(1).unsqueeze(2).expand(predictions.size(0), predictions.size(1), 1)
        targets = self._create_targets(targets)

        # reshape prediction to match grid
        predictions = predictions.reshape(predictions.size(0), predictions.size(1),
                                          len(self.elevation_angles), len(self.azimuth_angles))

        loss = self.criterion(predictions, targets)
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        # Update model
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.step += 1

        # Get values
        loss = loss.item() * self.gradient_accumulation_steps

        return loss

    def train_one_epoch(self) -> None:
        """
        Train one epoch.
        :return: None
        """
        self.model.train()

        train_loss = 0.0

        total_loss = 0.0

        batch_idx = None
        batch_start = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            # Save data loading wait time
            data_load_time = time.time() - batch_start

            # Train one batch
            loss = self.train_one_batch(batch_idx, data)

            # Accumulate scores
            train_loss += loss

            # Time batch
            batch_time = time.time() - batch_start

            # Log info
            if (floor(batch_idx + 1 / self.gradient_accumulation_steps) % self.log_interval == 0) or (batch_idx == 0):
                # Accumulate scores
                train_loss /= self.gradient_accumulation_steps

                total_loss += train_loss

                logger.info(f"Train Epoch {self.epoch} - Step {self.step}: "
                            f"Loss = {train_loss:.3f}, "
                            f"Batch Time = {batch_time:.3f}, "
                            f"Data Load Time = {data_load_time:.3f}, "
                            f"Learning Rate = {self.optimizer.param_groups[0]['lr']:.3e}")
                if self.use_wandb:
                    wandb.log(data={"train_loss": train_loss,
                                    "train_batch_time": batch_time,
                                    "train_data_load_time": data_load_time,
                                    "lr": self.optimizer.param_groups[0]["lr"]},
                              step=self.step)
                train_loss = 0.

            # Validate if necessary
            if self.validation_interval:
                if floor(batch_idx + 1 / self.gradient_accumulation_steps) % self.validation_interval == 0:
                    self.validate()

            # Time start of next batch
            batch_start = time.time()


        # Finished epoch
        avg_loss = total_loss / (batch_idx + 1)

        logger.info(f"Average Training Loss: {avg_loss:.3f}")
        if self.use_wandb:
            wandb.log({"avg_train_loss": avg_loss,
                       "epoch": self.epoch},
                      step=self.step)

        val_loss = self.validate()
        # Check if current validation loss is the best we've seen so far
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_model("best_model.pth")

    def validate(self):
        """
        Validate the model on the validation dataset.
        :return: avg_loss: float
        """
        if self.val_loader is None:
            logger.info("Validation data not provided. Skipping validation.")
            return None

        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            batch_idx = None
            for batch_idx, data in enumerate(self.val_loader):
                input_data, lengths, targets = data
                input_data, lengths, targets = (input_data.to(self.device),
                                                lengths.to(self.device),
                                                targets.to(self.device))

                # Forward pass
                predictions = self.model(input_data, lengths)

                # Create targets
                targets = targets.unsqueeze(1).unsqueeze(2).expand(predictions.size(0), predictions.size(1), 1)
                targets = self._create_targets(targets)

                # Reshape prediction to match target grid
                predictions = predictions.reshape(predictions.size(0), predictions.size(1),
                                                  len(self.elevation_angles), len(self.azimuth_angles))

                # Compute loss
                loss = self.criterion(predictions, targets)

                # Accumulate
                total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        if self.use_wandb:
            wandb.log(data={"avg_val_loss": avg_loss},
                      step=self.step)
        logger.info(f"Average Validation Loss: {avg_loss:.3f}")
        return avg_loss

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            if self.val_loader is not None:
                self.validate()
            self.save_checkpoint()

    def save_model(self, filename) -> None:
        save_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Checkpoint loaded from {self.checkpoint_path}, resuming from epoch {self.start_epoch}")


def setup_trainer(config, model, train_loader, val_loader):
    return ExampleTrainer(config, model, train_loader, val_loader)
