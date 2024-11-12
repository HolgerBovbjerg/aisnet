import os
import time
from math import ceil, floor
import logging
import datetime

import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from source.utils import get_device
from source.loss import get_loss
from source.metrics import equal_error_rate
from source.optimizer import get_optimizer
from source.scheduler import get_scheduler

logger = logging.getLogger(__name__)


def transform_speaker_ids_to_labels(speaker_ids_tensor, label_mapping):
    """Transforms a tensor of speaker IDs into corresponding class labels."""
    # Create a mapping tensor based on the provided dictionary
    # Assuming label_mapping is in the format {speaker_id: class_label}
    max_speaker_id = max(label_mapping.keys())

    # Create a tensor to hold the class labels, initializing with -1 (or any invalid label)
    label_tensor = torch.full((max_speaker_id + 1,), -1, dtype=torch.long)

    # Populate the tensor with the class labels based on the label mapping
    for speaker_id, class_label in label_mapping.items():
        label_tensor[speaker_id] = class_label

    # Use the label tensor to transform the input speaker_ids_tensor
    transformed_labels = label_tensor[speaker_ids_tensor]

    return transformed_labels


class ExampleTrainer:
    """
    Example Trainer class for training encoder.
    """

    def __init__(self,
                 config: DictConfig,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 label_mapping: dict,
                 distributed: bool = False):

        # Experiment settings
        self.distributed = distributed
        if self.distributed:
            self.rank = int(os.environ["SLURM_PROCID"])
            gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
            self.local_rank = self.rank - gpus_per_node * (self.rank // gpus_per_node)

        self.seed = config.training.seed
        self.device = self.local_rank if distributed else get_device(config.job.device.name)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        self.save_dir = os.path.join(config.job.exp_dir, config.job.exp_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logger.info('Saving results in %s.', self.save_dir)
        # self.wandb_id = None

        # Training settings
        self.num_epochs = config.training.num_epochs
        self.clip_grad_norm = config.training.clip_grad_norm
        if self.clip_grad_norm:
            logger.info('Clipping gradient norm at %s.', self.clip_grad_norm)
        self.gradient_accumulation_steps = 1 if config.training.gradient_accumulation_steps is None else config.training.gradient_accumulation_steps
        if self.gradient_accumulation_steps > 1:
            logger.info('Using %s gradient accumulation steps. Effective batch size will be %s times larger.',
                        self.gradient_accumulation_steps, self.gradient_accumulation_steps)
        self.use_cuda_amp = config.training.use_cuda_amp # whether Autmotaic Mixed Precision is used
        if self.use_cuda_amp:
            if torch.cuda.is_available():
                self.scaler = torch.amp.GradScaler(device="cuda")
            else:
                logger.info("'use_cuda_amp is set to True but CUDA is not available. Reverting training without AMP.")
                self.use_cuda_amp = False

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Label mapping
        self.label_mapping = label_mapping

        # Send model to device
        self.model = model
        self.model.to(self.device)
        if distributed:
            # Distributes model across GPUs using DistributedDataParallel.
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])

        # Set up the optimizer
        optimizer_config = OmegaConf.to_object(config.optimizer)
        logger.info(f"Optimizer config: {optimizer_config}")
        self.optimizer = get_optimizer(self.model, opt_config=optimizer_config)

        # Set up the loss function
        loss_config = OmegaConf.to_object(config.loss)
        logger.info(f"Loss config: {loss_config}")
        self.criterion = get_loss(**loss_config)

        # Setup lr scheduler
        if config.scheduler is not None:
            self.scheduler = self.setup_scheduler(config)
        else:
            self.scheduler = None

        # Job settings
        self.log_interval = config.job.log_interval  # Log every N batches
        self.validation_interval = config.job.validation_interval  # Validate every N batches
        # only used if smaller than number of batches. Ignored if None.

        # Initialize training variables
        self.start_epoch = 0
        self.epoch = 0
        self.steps = 0
        self.batch_index = 0
        self.best_val_score = float('inf')
        self.last_log_step = 0
        self.last_validation_step = 0
        self.accumulated_loss = 0.0
        self.accumulated_steps = 0
        self.total_epoch_loss = 0.0
        self.batch_time = 0.
        self.data_load_time = 0.

        # Check for existing checkpoint
        self.checkpoint_path = os.path.join(self.save_dir, "checkpoint.pth")
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()
        if distributed:
            torch.distributed.barrier()

        # Wandb settings
        self.use_wandb = config.wandb.enabled
        if self.use_wandb:
            #if self.wandb_id is None:
                #self.wandb_id = wandb.util.generate_id()
            self.setup_wandb(config)

    def setup_wandb(self, config):
        if self.distributed:
            if self.rank != 0:
                return None
        wandb.init(project=config.wandb.project,
                   entity=config.wandb.entity,
                   name=config.wandb.name,
                   group=config.wandb.group,
                   job_type=config.wandb.job_type,
                   tags=config.wandb.tags)
                   #resume=config.wandb.resume)
                   #id=self.wandb_id)
        wandb.config.update(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))

    def setup_scheduler(self, config):
        scheduler_config = OmegaConf.to_object(config.scheduler)
        logger.info(f"Scheduler config: {scheduler_config}")
        if config.scheduler.scheduler_type is not None:
            if config.scheduler.steps_per_epoch is not None:
                total_iters = config.scheduler.steps_per_epoch * max(1, config.training.num_epochs)
                scheduler = get_scheduler(self.optimizer,
                                          scheduler_type=scheduler_config["scheduler_type"],
                                          t_max=total_iters,
                                          **scheduler_config["scheduler_kwargs"])
            else:
                steps_per_epoch = ceil(len(self.train_loader) / config.training.gradient_accumulation_steps)
                total_iters = steps_per_epoch * max(1, config.training.num_epochs)

                scheduler = get_scheduler(self.optimizer,
                                          scheduler_type=scheduler_config["scheduler_type"],
                                          t_max=total_iters,
                                          **scheduler_config["scheduler_kwargs"])
        else:
            logger.error("'scheduler_type' not specified, defaulting to 'None'")
            scheduler = None

        return scheduler

    def log(self):
        train_loss = self.accumulated_loss / self.accumulated_steps
        logger.info(f"Train Epoch {self.epoch} - Step {self.steps} - Batch Index {self.batch_index}: "
                    f"Loss = {train_loss:.3f}, "
                    f"Batch Time = {self.batch_time:.3f}, "
                    f"Data Load Time = {self.data_load_time:.3f}, "
                    f"Learning Rate = {self.optimizer.param_groups[0]['lr']:.3e}")
        if self.use_wandb:
            wandb.log(data={"train_loss": train_loss,
                            "train_batch_time": self.batch_time,
                            "train_data_load_time": self.data_load_time,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "batch_index": self.batch_index},
                      step=self.steps)
        self.accumulated_loss = 0.0
        self.accumulated_steps = 0
        self.last_log_step = self.steps

    def forward_pass(self, data):
        # Unpack data
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

        return loss

    def backward_pass(self, loss):
        if self.use_cuda_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def update_model(self):
        if ((self.batch_index + 1) % self.gradient_accumulation_steps) == 0:
            if self.use_cuda_amp:
                self.scaler.unscale_(self.optimizer)
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.step()

    def train_one_batch(self, data):
        """
        Trains the model for one batch.
        :param batch_index: int
        :param data: tuple of data from data loader
        :return: loss: float
        """

        # CUDA mixed precision context manager, AMP is disabled if self.use_cuda_amp=False
        with torch.autocast(device_type="cuda", enabled=self.use_cuda_amp):
            # Forward pass
            loss = self.forward_pass(data)
            # Divide loss by number of gradient accumulation steps
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

        # Backward pass
        self.backward_pass(loss)

        # Update model
        self.update_model()

        # Get batch loss value without after unscaling by gradient_accumulation_steps
        loss = loss.item()
        if self.gradient_accumulation_steps > 1:
            loss *= self.gradient_accumulation_steps
        return loss

    def step(self):
        if self.use_cuda_amp:
            self.scaler.step(self.optimizer)  # Update weights
            self.scaler.update()  # Update scaler for next step
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad()
        self.steps += 1

    def train_one_epoch(self) -> None:
        """
        Train one epoch.
        :return: None
        """
        self.model.train()
        self.total_epoch_loss = 0.0
        self.accumulated_loss = 0.0
        self.accumulated_steps = 0

        batch_start_time = time.time()
        for batch_index, data in enumerate(self.train_loader):
            self.batch_index = batch_index
            # Save data loading wait time
            self.data_load_time = time.time() - batch_start_time

            # Train one batch
            loss = self.train_one_batch(data)

            # Accumulate scores
            self.accumulated_loss += loss
            self.accumulated_steps += 1
            self.total_epoch_loss += loss

            # Time batch
            self.batch_time = time.time() - batch_start_time

            # Log info
            if ((self.steps % self.log_interval) == 0) and (self.steps != self.last_log_step):
                self.log()

            # Validate if necessary
            if self.validation_interval:
                if ((self.steps % self.validation_interval) == 0) and (self.steps != self.last_validation_step):
                    if self.distributed:
                        if self.rank == 0:
                            self.validate()
                    else:
                        self.validate()

            # Time start of next batch
            batch_start_time = time.time()

        # Log training metrics if any accumulated results remain
        if self.accumulated_steps > 0:
            self.log()

        # Log average metrics
        avg_loss = self.total_epoch_loss / (self.batch_index + 1)
        logger.info(f"Average Training Loss: {avg_loss:.3f}")
        if self.use_wandb:
            wandb.log({"avg_train_loss": avg_loss,
                       "epoch": self.epoch},
                      step=self.steps)

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

                eer = equal_error_rate(predictions.numpy(), targets.numpy())

                # Accumulate
                total_eer += eer

        avg_eer = total_eer / (batch_index + 1)
        if self.use_wandb:
            wandb.log(data={"avg_val_eer": avg_eer},
                      step=self.steps)
        self.last_validation_step = self.steps
        logger.info(f"Average Validation EER: {avg_eer:.3f}")
        return avg_eer

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            self.epoch = epoch
            self.train_one_epoch()
            if self.distributed:
                if self.rank == 0:
                    val_score = self.validate()
                    # Check if current validation loss is the best we've seen so far
                    if val_score < self.best_val_score:
                        self.best_val_score = val_score
                        self.save_model("best_model.pth")
            else:
                val_score = self.validate()
                # Check if current validation loss is the best we've seen so far
                if val_score < self.best_val_score:
                    self.best_val_score = val_score
                    self.save_model("best_model.pth")

            # Finished epoch
            if self.distributed:
                if self.rank != 0:
                    return None
            epoch_time = time.time() - epoch_start_time
            self.end_of_epoch(epoch, epoch_time)

    def end_of_epoch(self, epoch, epoch_time):
        if self.use_wandb:
            wandb.log(data={"epoch_time": epoch_time},
                      step=self.steps)

        estimated_time_to_finish = (self.num_epochs - (epoch+1)) * epoch_time
        hours = int(estimated_time_to_finish // 3600)
        minutes = int((estimated_time_to_finish % 3600) // 60)
        seconds = int(estimated_time_to_finish % 60)
        logger.info(f"Finished Epoch {epoch} "
                    f"- Estimated time to finish: {hours}h {minutes}m {seconds}s")
        self.save_checkpoint()

    def save_model(self, filename) -> None:
        if self.distributed:
            if self.rank == 0:
                save_path = os.path.join(self.save_dir, filename)
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Model saved to {save_path}")
        else:
            save_path = os.path.join(self.save_dir, filename)
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    def save_checkpoint(self):
        if self.distributed:
            if self.rank == 0:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'epoch': self.epoch,
                    'best_val_score': self.best_val_score
                }
                torch.save(checkpoint, self.checkpoint_path)
                logger.info(f"Checkpoint saved to {self.checkpoint_path}")
        else:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': self.epoch,
                'best_val_score': self.best_val_score,
                'steps': self.steps
            }
            #if self.wandb_id:
                #checkpoint["wandb_id"] = self.wandb_id
            torch.save(checkpoint, self.checkpoint_path)
            logger.info(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_score = checkpoint['best_val_score']
        self.steps = checkpoint['steps']
        #if "wandb_id" in list(checkpoint.keys()):
            #self.wandb_id = checkpoint["wandb_id"]
        logger.info(f"Checkpoint loaded from {self.checkpoint_path}, resuming from epoch {self.start_epoch}")


def setup_trainer(config, model, train_loader, val_loader, label_mapping, distributed):
    return ExampleTrainer(config, model, train_loader, val_loader, label_mapping, distributed)
