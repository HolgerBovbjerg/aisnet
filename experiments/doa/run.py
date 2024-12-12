from logging import getLogger
import os
from socket import gethostname

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from source.utils import seed_everything
from source.models.utils import summarize_model, print_module_structure
from .data_preparation import prepare_data
from .data_loading import setup_dataloader
from .model import build_model
from .trainer import setup_trainer
from .evaluator import setup_evaluator

logger = getLogger(__name__)


def distributed_setup(backend: str = "nccl"):
    logger.info("Setting up distributed process group...")
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    logger.info(f"Setting up distributed process group on {rank} of {world_size} on {gethostname()} where there are"
                f" {gpus_per_node} allocated GPUs per node.")
    init_process_group(rank=rank, world_size=world_size, backend=backend)
    dist.barrier()
    if rank == 0:
        logger.info(f"Group initialized? {dist.is_initialized()}")


def distributed_cleanup():
    logger.info("Destroying all distributed processes and cleaning up.")
    destroy_process_group()

def model_training(config, distributed: bool = False):
    if distributed:
        # Setup: Initializes the distributed environment.
        distributed_setup(backend=config.job.device.backend)
    logger.info("Setting seed.")
    seed_everything(config.training.seed)
    logger.info("Setting up data loaders.")
    train_loader, val_loader = setup_dataloader(config, distributed=distributed)
    logger.info("Building model.")
    model = build_model(config)
    logger.info("Built model:")
    print_module_structure(model)
    summarize_model(model)
    logger.info("Setting up trainer.")
    trainer = setup_trainer(config, model, train_loader, val_loader, distributed=distributed)
    logger.info("Starting training.")
    if distributed:
        dist.barrier()
    trainer.train()
    logger.info("Finished training.")

    if distributed:
        # Cleanup: Releases distributed training resources after completion.
        distributed_cleanup()


def run(config, distributed: bool = False):

    start_stage = config.start_stage
    stop_stage = config.stop_stage
    stage = start_stage

    while stage <= stop_stage:
        if stage == 0:
            logger.info("Stage {}: Data preparation:".format(stage))
            prepare_data(config)
            stage += 1
        if stage == 1:
            logger.info("Stage {}: Model training".format(stage))
            if distributed:
                logger.info("Launching distributed training job.")
                world_size = int(os.environ['WORLD_SIZE'])
                mp.spawn(model_training, args=(config, distributed), nprocs=world_size, join=True)
            else:
                logger.info("Launching local job.")
                model_training(config)
            stage += 1
        if stage == 2:
            logger.info("Stage {}: Evaluating trained model".format(stage))
            evaluator = setup_evaluator(config, distributed=distributed)
            evaluator.evaluate()
            logger.info("All stages finished.")
            break

    logger.info("Experiment done.")
