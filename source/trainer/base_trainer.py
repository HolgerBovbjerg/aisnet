import os
import time
from math import ceil
import logging

import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.distributed as dist

from source.utils import get_device
from source.loss import get_loss
from source.optimizer import get_optimizer
from source.scheduler import get_scheduler

logger = logging.getLogger(__name__)


def reduce_metric(metric):
    # Ensure `metric` is a tensor for reduction
    metric_tensor = torch.tensor(metric).cuda()
    dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.SUM)
    metric_tensor /= dist.get_world_size()
    return metric_tensor.item()


class BaseTrainer:
    """
    Base Trainer class for training PyTorch models.
    """

    def __init__(self,
                 config: DictConfig,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
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
        self.device = self.local_rank if distributed else get_device(config.job.device.name)
        if torch.cuda.is_available() and distributed:
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
                logger.info("'use_cuda_amp is set to True but CUDA is not available. Reverting to training without AMP.")
                self.use_cuda_amp = False
        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_min_epochs = config.training.early_stopping_min_epochs

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

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

        # Setup metrics
        self.metrics_to_log = ["loss"]

        # Job settings
        self.log_interval = config.job.log_interval  # Log every N batches
        self.validation_interval = config.job.validation_interval  # Validate every N batches
        # only used if smaller than number of batches. Ignored if None.

        # Initialize training variables
        self.start_epoch = 0
        self.total_time = 0.0
        self.epoch = 0
        self.steps = 0
        self.batch_steps = 0
        self.batch_index = 0
        self.best_val_score = float('inf')
        self.last_log_step = 0
        self.last_validation_step = 0
        self.total_epoch_loss = 0.0
        self.batch_time = 0.
        self.data_load_time = 0.
        self.epochs_no_improve = 0
        self.stop_training = False

        # Initialize with default metric: loss
        self.accumulated_metrics = {"loss": 0.0}
        self.accumulated_steps = 0
        self.total_epoch_metrics = {"loss": 0.0}
        self.validation_metrics = {"loss": 0.0}

        # Check for existing checkpoint
        self.checkpoint_path = os.path.join(self.save_dir, "checkpoint.pth")
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()
        if distributed:
            dist.barrier()

        # Wandb settings
        self.use_wandb = config.wandb.enabled
        if self.use_wandb:
            #if self.wandb_id is None:
                #self.wandb_id = wandb.util.generate_id()
            self.setup_wandb(config)

    def initialize_metrics(self):
        # Initialize with any default metric, e.g., loss
        self.accumulated_metrics = {metric: 0.0 for metric in self.metrics_to_log}
        self.accumulated_steps = 0
        self.total_epoch_metrics = {metric: 0.0 for metric in self.metrics_to_log}
        self.validation_metrics = {metric: 0.0 for metric in self.metrics_to_log}

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
        # Calculate and log each metric in accumulated metrics
        avg_metrics = {metric: value / self.accumulated_steps for metric, value in self.accumulated_metrics.items()}

        # Synchronize if using distributed training
        if self.distributed:
            avg_metrics = {metric: reduce_metric(value) for metric, value in avg_metrics.items()}

        if self.rank == 0:
            # Construct the log message dynamically based on available metrics
            log_message = (f"Train Epoch {self.epoch} - Step {self.steps} - Batch Index {self.batch_index}: "
                           + ", ".join([f"{metric.lower()} = {avg_metrics[metric]:.3f}" for metric in avg_metrics])
                           + f", batch time = {self.batch_time:.3f}, data load time = {self.data_load_time:.3f}, "
                             f"learning rate = {self.optimizer.param_groups[0]['lr']:.3e}")

            logger.info(log_message)
            # Log metrics to Weights & Biases if enabled
            if self.use_wandb:
                # Combine metrics into a dictionary with additional batch-specific information
                wandb_data = {f"train_{metric}": avg_metrics[metric] for metric in avg_metrics}
                wandb_data.update({
                    "train_batch_time": self.batch_time,
                    "train_data_load_time": self.data_load_time,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "batch_index": self.batch_index
                })
                wandb.log(data=wandb_data, step=self.steps)
        self.reset_metrics()
        self.last_log_step = self.steps

    def reset_metrics(self):
        """
        Reset metrics for a new accumulation period.
        """
        self.accumulated_metrics = {metric: 0.0 for metric in self.metrics_to_log}
        self.accumulated_steps = 0

    def backward_pass(self, loss):
        """
        Performs backward pas
        :param loss: loss from model
        :return: None
        """
        if self.use_cuda_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def update_model(self):
        """
        Updates model parameters and scheduler
        :return: None
        """
        if ((self.batch_index + 1) % self.gradient_accumulation_steps) == 0:
            if self.use_cuda_amp:
                self.scaler.unscale_(self.optimizer)
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.step()

    def step(self):
        """
        Takes a single optimizer and scheduler step
        :return: None
        """
        if self.use_cuda_amp:
            self.scaler.step(self.optimizer)  # Update weights
            self.scaler.update()  # Update scaler for next step
        else:
            self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        self.optimizer.zero_grad()
        self.steps += 1
        self.batch_steps += 1

    def train_one_epoch(self) -> None:
        """
        Train model for one epoch.
        :return: None
        """
        self.model.train()
        self.total_epoch_metrics = {metric: 0.0 for metric in self.total_epoch_metrics}
        self.accumulated_metrics = {metric: 0.0 for metric in self.accumulated_metrics}
        self.accumulated_steps = 0

        batch_start_time = time.time()
        self.batch_steps = 0
        for batch_index, data in enumerate(self.train_loader):
            # Save data loading wait time
            self.data_load_time = time.time() - batch_start_time

            # Set batch index
            self.batch_index = batch_index

            # Train one batch
            batch_metrics = self.train_one_batch(data)

            # Accumulate metrics
            for metric, value in batch_metrics.items():
                self.accumulated_metrics[metric] += value
                self.total_epoch_metrics[metric] += value
            self.accumulated_steps += 1

            # Time batch
            self.batch_time = time.time() - batch_start_time

            # Log info
            if ((self.batch_steps % self.log_interval) == 0) and (self.steps != self.last_log_step):
                self.log()

            # Validate model
            if (self.validation_interval
                    and ((self.steps % self.validation_interval) == 0)
                    and (self.steps != self.last_validation_step)
                    and (self.rank == 0)):
                self.validate()

            # Time start of next batch
            batch_start_time = time.time()
        
        # Log training metrics if any accumulated results remain
        if self.accumulated_steps > 0:
            self.log()

        # Log average metrics
        avg_epoch_metrics = {metric: value / (self.batch_index + 1) for metric, value in self.total_epoch_metrics.items()}
        # Construct a log message with formatted metric values
        avg_epoch_metrics_message = ", ".join(
            [f"{metric.lower()} = {avg_epoch_metrics[metric]:.3f}" for metric in avg_epoch_metrics])
        logger.info(f"Average Training Metrics - Epoch {self.epoch}: {avg_epoch_metrics_message}")
        # Log average metrics to wandb
        if self.use_wandb:
            avg_metrics_wandb = {"train_" + metric: value for metric, value in avg_epoch_metrics.items()}
            wandb.log({**avg_metrics_wandb, "epoch": self.epoch}, step=self.steps)

    def train(self) -> None:
        epoch_start_time = time.time()
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.distributed:
                dist.barrier()
            self.epoch = epoch
            self.train_one_epoch()
            if self.distributed:
                dist.barrier()
            if self.rank == 0:
                val_score = self.validate()
                # Check if current validation loss is the best we've seen so far
                if val_score < self.best_val_score:
                    self.best_val_score = val_score
                    self.save_model("best_model.pth")
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
            self.end_of_epoch(epoch, epoch_start_time)
            if self.stop_training:
                if self.rank == 0:
                    logger.info("Validation score not improved for %s epochs. "
                                "Early stopping training at epoch %s", self.early_stopping_patience, self.epoch)
                break
            epoch_start_time = time.time()
        if self.distributed:
            dist.barrier()
        if self.rank == 0:
            total_time = self.total_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            logger.info("Finished training for %s epochs, starting from epoch %s. Training complete.",
                        self.epoch + 1, self.start_epoch)
            logger.info("Total training time: %sh %sm %ss",
                        hours, minutes, seconds)

    def end_of_epoch(self, epoch, epoch_start_time):
        # Check early stopping criterion
        if ((self.epochs_no_improve >= self.early_stopping_patience)
                and ((self.epoch + 1) > self.early_stopping_min_epochs)):
            self.stop_training = True

        # Broadcast stop_training to all workers so they also terminate if necessary
        if self.distributed:
            # Convert self.stop_training to a tensor and broadcast
            stop_signal = torch.tensor(int(self.stop_training), device=self.device)
            dist.broadcast(stop_signal, src=0)
            # Update self.stop_training for all ranks based on broadcasted value
            self.stop_training = stop_signal.item() == 1

        # Synchronize all processes before proceeding to next epoch
        if self.distributed:
            dist.barrier()

        # Log epoch time and save checkpoint on rank 0 worker
        if self.rank == 0:
            epoch_time = time.time() - epoch_start_time
            self.total_time += epoch_time
            if self.use_wandb:
                wandb.log(data={"epoch_time": epoch_time},
                          step=self.steps)

            estimated_time_to_finish = (self.num_epochs - (epoch + 1)) * epoch_time
            hours = int(estimated_time_to_finish // 3600)
            minutes = int((estimated_time_to_finish % 3600) // 60)
            seconds = int(estimated_time_to_finish % 60)
            logger.info("Finished Epoch %s "
                        "- Estimated time to finish: %sh %sm %ss", epoch, hours, minutes, seconds)
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
                    'best_val_score': self.best_val_score,
                    'steps': self.steps,
                    'total_time': self.total_time
                }
                if self.use_cuda_amp:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                torch.save(checkpoint, self.checkpoint_path)
                logger.info(f"Checkpoint saved to {self.checkpoint_path}")
        else:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': self.epoch,
                'best_val_score': self.best_val_score,
                'steps': self.steps,
                'total_time': self.total_time
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
        self.total_time = checkpoint['total_time']
        if self.use_cuda_amp and 'scaler' in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint['scaler'])
            except (KeyError, RuntimeError) as e:
                logger.error(
                    f"Failed to load scaler state from checkpoint: {e}. Using a new scaler initialized from scratch.")
        #if "wandb_id" in list(checkpoint.keys()):
            #self.wandb_id = checkpoint["wandb_id"]
        logger.info(f"Checkpoint loaded from {self.checkpoint_path}, resuming from epoch {self.start_epoch}")

    def forward_pass(self, data):
        # Unpack data
        input_data, lengths, targets = data
        # Transfer data to device
        input_data, lengths, targets = (input_data.to(self.device),
                                        lengths.to(self.device),
                                        targets.to(self.device))
        # Make prediction
        predictions = self.model(input_data, lengths=lengths)
        # Compute loss
        loss = self.criterion(predictions, targets)
        return loss, predictions, targets

    def compute_batch_metrics(self, loss, predictions, targets):
        metrics = {"loss": loss}
        if self.distributed:
            metrics = {metric: reduce_metric(value) for metric, value in metrics.items()}
        return metrics

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
            loss, predictions, targets = self.forward_pass(data)
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
        return self.compute_batch_metrics(loss, predictions.detach(), targets.detach())


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
        with torch.no_grad():
            total_loss = 0.0
            batch_index = 0
            for batch_index, data in enumerate(self.val_loader):
                # Unpack data
                input_data, lengths, targets = data
                # Transfer data to device
                input_data, lengths, targets = (input_data.to(self.device),
                                                lengths.to(self.device),
                                                targets.to(self.device))
                # Make prediction
                predictions = self.model(input_data, lengths=lengths)
                # Compute loss
                loss = self.criterion(predictions, targets)
                # Accumulate
                total_loss += loss

        avg_loss = total_loss / (batch_index + 1)
        if self.use_wandb:
            wandb.log(data={"avg_val_loss": avg_loss},
                      step=self.steps)
        self.last_validation_step = self.steps
        logger.info(f"Average Validation Loss: {avg_loss:.3f}")
        return avg_loss
