import os
import time
from math import ceil

import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from common.misc import seed_everything
from common.loss import get_loss
from common.optimizer import get_optimizer
from common.scheduler import get_scheduler
from .BPCTrainer import BPCTrainer


class BPCDistributedTrainer(BPCTrainer):
    """
    BPC Trainer class for training BPC nnet.
    """

    def __init__(self,
                 config: DictConfig,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 rank: int):
        super().__init__(config=config, model=model, train_loader=train_loader, val_loader=val_loader)

        # Experiment settings
        self.device = rank

        # DDP
        torch.cuda.set_device(self.device)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        # Send model to device
        self.model = DDP(model, device_ids=[self.device])
        self.model.to(self.device)

        # Wandb settings
        if rank == 0:
            self.use_wandb = config.wandb.enabled
            if self.use_wandb:
                wandb.init(project=config.wandb.project,
                           entity=config.wandb.entity,
                           name=config.wandb.name,
                           group=config.wandb.group,
                           job_type=config.wandb.job_type,
                           tags=config.wandb.tags)
                wandb.config.update(OmegaConf.to_container(config))

        # Initialize training variables
        self.epoch = None
        self.start_epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')

    def train_one_batch(self, batch_idx, data):
        """
        Trains the model for one batch.
        :param batch_idx: int
        :param data: tuple of data from data loader
        :return: (loss: float, target_var: float, prediction_var: float)
        """
        noisy_waveforms_padded, clean_waveforms_padded, lengths = data
        noisy_waveforms_padded, clean_waveforms_padded, lengths = (noisy_waveforms_padded.to(self.device),
                                                                   clean_waveforms_padded.to(self.device),
                                                                   lengths.to(self.device))

        self.optimizer.zero_grad()

        # Forward pass
        prediction, target = self.model(teacher_input=clean_waveforms_padded, student_input=noisy_waveforms_padded,
                                        lengths=lengths, mask=True)

        # Compute loss and gradients
        loss = self.criterion(prediction, target)
        loss.backward()
        loss = loss / self.loss_accumulation_steps
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        # Update model
        if (batch_idx + 1) % self.loss_accumulation_steps == 0:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.model.ema_step()
            self.step += 1

        # Compute prediction variance
        target_var = target.std(dim=-1).mean()
        prediction_var = prediction.std(dim=-1).mean()

        # Get values
        loss = loss.item()
        target_var = target_var.item()
        prediction_var = prediction_var.item()

        return loss, target_var, prediction_var

    def train_one_epoch(self) -> None:
        """
        Train one epoch.
        :return: None
        """
        self.model.train()
        total_loss = 0.0
        total_target_var = 0.0
        total_prediction_var = 0.0
        batch_idx = None
        batch_start = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            # Save data loading wait time
            data_load_time = time.time() - batch_start

            # Train one batch
            loss, target_var, prediction_var = self.train_one_batch(batch_idx, data)

            # Accumulate scores
            total_loss += loss
            total_target_var += target_var
            total_prediction_var += prediction_var

            # Time batch
            batch_time = time.time() - batch_start

            # Log info
            if self.device == 0 and batch_idx % self.log_interval:
                print(f"Train Epoch {self.epoch} - Batch {batch_idx}: Loss = {loss:.3f}"
                      f", Target Var = {target_var:.3f}", f", Prediction Var = {prediction_var:.3f}"
                                                          f", Batch Time = {batch_time:.3f}, Data Load Time = {data_load_time:.3f}"
                                                          f", Learning Rate = {self.optimizer.param_groups[0]['lr']:.3e}")
                if self.use_wandb:
                    wandb.log(data={"train_loss": loss,
                                    "train_target_var": target_var,
                                    "train_prediction_var": prediction_var,
                                    "train_batch_time": batch_time,
                                    "train_data_load_time": data_load_time,
                                    "lr": self.optimizer.param_groups[0]["lr"]},
                              step=self.step)
                if self.validation_interval is not None:
                    if batch_idx % self.validation_interval == 0:
                        self.validate()

            batch_start = time.time()

        # Finished epoch
        avg_loss = total_loss / (batch_idx + 1)
        avg_target_var = total_target_var / (batch_idx + 1)
        avg_prediction_var = total_prediction_var / (batch_idx + 1)

        if self.device == 0:
            print(f"Average Training Loss: {avg_loss:.3f}"
                  f", Average Training Target Var: {avg_target_var:.3f}"
                  f", Average Training Prediction Var: {avg_prediction_var:.3f}")
            if self.use_wandb:
                wandb.log({"avg_train_loss": avg_loss,
                           "avg_train_target_var": avg_target_var,
                           "avg_train_prediction_var": avg_prediction_var,
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
            print("Validation data not provided.")
            return None

        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_target_var = 0.0
            total_prediction_var = 0.0
            batch_idx = None
            for batch_idx, data in enumerate(self.val_loader):
                noisy_waveforms_padded, clean_waveforms_padded, lengths = data
                noisy_waveforms_padded, clean_waveforms_padded, lengths = (noisy_waveforms_padded.to(self.device),
                                                                           clean_waveforms_padded.to(self.device),
                                                                           lengths.to(self.device))

                # Forward pass
                prediction, target = self.model(teacher_input=clean_waveforms_padded,
                                                student_input=noisy_waveforms_padded,
                                                lengths=lengths, mask=True)

                # Compute loss
                loss = self.criterion(prediction, target)

                # Compute prediction variance
                target_var = target.std(dim=-1).mean()
                prediction_var = prediction.std(dim=-1).mean()

                # Accumulate
                total_loss += loss.item()
                total_target_var += target_var.item()
                total_prediction_var += prediction_var.item()

        avg_loss = total_loss / (batch_idx + 1)
        avg_target_var = total_target_var / (batch_idx + 1)
        avg_prediction_var = total_prediction_var / (batch_idx + 1)

        if self.device == 0:
            print(f"Average Validation Loss: {avg_loss:.3f}"
                  f", Average Validation Target Var: {avg_target_var:.3f}"
                  f", Average Validation Prediction Var: {avg_prediction_var:.3f}")
            if self.use_wandb:
                wandb.log(data={"avg_val_loss": avg_loss,
                                "avg_val_target_var": avg_target_var,
                                "avg_val_prediction_var": avg_prediction_var},
                          step=self.step)

        return avg_loss

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch
            self.train_sampler.set_epoch(epoch)
            self.train_one_epoch()
            if self.val_loader is not None:
                self.validate()
            self.save_checkpoint()