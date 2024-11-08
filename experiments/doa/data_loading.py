from logging import getLogger
from typing import Optional

from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.nn.utils.rnn import pad_sequence

from source.datasets import BinauralLibriSpeechDataset, get_data_loader
from .augmentation import get_augmentor

logger = getLogger(__name__)


def pad_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data["waveform"].T for data in batch]
    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in waveforms]  # Instance norm
    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])
    doas = torch.tensor([data["azimuth"] for i, data in enumerate(batch)])

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0).transpose(-1, -2)

    return waveforms_padded, lengths, doas


def setup_dataloader(config, rank: Optional[int] = None, world_size: Optional[int] = None):
    logger.info("Creating data augmentor.")
    augmentor = get_augmentor(config)

    # Create datapipe and dataloaders
    logger.info("Creating datasets...")
    logger.info(f"Data Config: {OmegaConf.to_object(config.data)}")
    train_config = config.data.train
    train_dataset = BinauralLibriSpeechDataset(root_dir=train_config.root_dir,
                                               metadata_filename=train_config.metadata_filename,
                                               split=train_config.split)
    validation_config = config.data.validation
    validation_dataset = BinauralLibriSpeechDataset(root_dir=validation_config.root_dir,
                                                    metadata_filename=validation_config.metadata_filename,
                                                    split=validation_config.split)
    #
    logger.info("Creating data samplers...")
    if config.job.device == "distributed":
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
        val_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
    else:
        train_sampler = None
        val_sampler = None
    logger.info("Creating dataloaders...")
    train_loader = get_data_loader(train_dataset, batch_size=train_config.batch_size, shuffle=False,
                                   sampler=train_sampler, pin_memory=config.job.pin_memory,
                                   collate_fn=pad_collate)
    validation_loader = get_data_loader(validation_dataset, batch_size=validation_config.batch_size,
                                        shuffle=False, sampler=val_sampler,
                                        pin_memory=config.job.pin_memory,
                                        collate_fn=pad_collate)
    return train_loader, validation_loader
