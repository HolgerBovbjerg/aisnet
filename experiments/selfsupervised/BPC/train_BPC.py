import os
from typing import Optional
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

from BPC import BPCTrainer, BPCDistributedTrainer, BPC, BPCConfig, build_bpc_datapipe, get_data_loader
from common.augment import get_composed_augmentations
from common.misc import count_parameters


# Setting up the logger in the main script
logger = logging.getLogger(__name__)


def distributed_setup(rank: int, world_size: int, backend: str = "nccl"):
    logger.info("Setting up distributed process group...")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine
    init_process_group(rank=rank, world_size=world_size, backend=backend)


def distributed_cleanup():
    logger.info("Destroying all distributed processes and cleaning up.")
    destroy_process_group()


def get_model_and_dataloader(config: DictConfig, rank: Optional[int] = None, world_size: Optional[int] = None):
    # Create feature extractor
    logger.info("Creating feature extractor...")


    # Create BPC model with configuration
    logger.info("Creating BPC model...")
    logger.info(f"Model Config: {OmegaConf.to_object(config.model)}")
    model_config = BPCConfig(**config.model)
    model = BPC(cfg=model_config)
    logger.info(f"Created model with {count_parameters(model)} parameters.")

    # Create data augmentor
    logger.info("Creating data augmentor...")
    logger.info(f"Augmentor Config: {OmegaConf.to_object(config.augment.waveform)}")
    wav_augmentor = get_composed_augmentations(OmegaConf.to_object(config.augment.waveform))

    # Create datapipe and dataloaders
    logger.info("Creating datapipes...")
    logger.info(f"Data Config: {OmegaConf.to_object(config.data)}")
    train_datapipe = build_bpc_datapipe(**config.data.train, clean_and_noisy=True, augmentor=wav_augmentor)
    val_datapipe = build_bpc_datapipe(**config.data.validation, clean_and_noisy=True, augmentor=wav_augmentor)

    logger.info("Creating data samplers...")
    if config.job.device == "distributed":
        train_sampler = DistributedSampler(train_datapipe, rank=rank, num_replicas=world_size)
        val_sampler = DistributedSampler(train_datapipe, rank=rank, num_replicas=world_size)
    else:
        train_sampler = None
        val_sampler = None

    logger.info("Creating dataloaders...")
    train_loader = get_data_loader(train_datapipe, batch_size=1, shuffle=~(config.job.device == "distributed"),
                                   sampler=train_sampler, pin_memory=config.job.pin_memory)
    val_loader = get_data_loader(val_datapipe, batch_size=1, shuffle=False, sampler=val_sampler,
                                 pin_memory=config.job.pin_memory)

    logger.info("Finished creating nnet and dataloaders.")
    return model, train_loader, val_loader


@hydra.main(config_path="BPC/BPC_configs", config_name="BPC_default_config", version_base=None)
def main(cfg: DictConfig = None) -> None:
    """
    Main function for BPC training.
    :param cfg: BPC training config
    :return: None
    """

    # Create trainer and start training
    if cfg.job.device == "distributed":
        logger.info("Using distributed training.")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        distributed_setup(rank, world_size, backend="nccl")
        model, train_loader, val_loader = get_model_and_dataloader(cfg, rank, world_size)
        trainer = BPCDistributedTrainer(cfg, model, train_loader, val_loader, rank)
        trainer.train()
        distributed_cleanup()
    else:
        logger.info("Using single node training")
        model, train_loader, val_loader = get_model_and_dataloader(cfg)
        trainer = BPCTrainer(cfg, model, train_loader, val_loader)
        trainer.train()


if __name__ == "__main__":
    main()
