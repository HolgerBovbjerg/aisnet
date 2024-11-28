from logging import getLogger
from typing import Optional
import os

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, Sampler
from torch.nn.utils.rnn import pad_sequence

from .augmentation import get_augmentor
from source.datasets.BinauralDataloader import build_binaural_datapipe

logger = getLogger(__name__)


def create_dataset(config, augmentor):
    dataset = build_binaural_datapipe(**config,
                                      augmentor=augmentor)
    return dataset

def pad_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data[0].transpose(-1, -2) for data in batch[0]]

    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9)
                 for waveform in waveforms]  # Instance norm

    lengths = torch.tensor([waveform.size(-1) for waveform in waveforms])

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0).transpose(-1, -2)

    return waveforms_padded, lengths

def pad_collate_clean_noisy(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data[0].transpose(-1, -2) for data in batch[0]]
    noisy_waveforms = [data[1].transpose(-1, -2) for data in batch[0]]

    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9)
                 for waveform in waveforms]  # Instance norm
    noisy_waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9)
                       for waveform in noisy_waveforms]  # Instance norm

    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])

    clean_waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0).transpose(-1, -2)
    noisy_waveforms_padded = pad_sequence(noisy_waveforms, batch_first=True, padding_value=0).transpose(-1, -2)

    return noisy_waveforms_padded, lengths, clean_waveforms_padded


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=pad_collate_clean_noisy,
                    pin_memory: bool = False, sampler: Optional[Sampler] = None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, sampler=sampler)
    return data_loader


def setup_dataloader(config, distributed=False):
    logger.info("Creating data augmentor.")
    if config.augment:
        augmentor = get_augmentor(config)
    else:
        augmentor = None
    # Create datapipe and dataloaders
    logger.info("Creating datasets...")
    logger.info(f"Data Config: {OmegaConf.to_object(config.data)}")
    train_config = config.data.train
    validation_config = config.data.validation

    train_dataset= create_dataset(train_config, augmentor)
    validation_dataset = create_dataset(validation_config, augmentor)

    logger.info("Creating data samplers...")
    if distributed:
        # Distributed Sampler: Ensures data is divided among GPUs using DistributedSampler.
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
        validation_sampler = DistributedSampler(validation_dataset, rank=rank, num_replicas=world_size)
    else:
        train_sampler = None
        validation_sampler = None

    logger.info("Creating dataloaders...")
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"] if distributed else config.job.num_workers)

    collate_fns = {
        "pad_collate_clean_noisy": pad_collate_clean_noisy,
        "pad_collate": pad_collate,
    }
    try:
        collate_fn = collate_fns[config.data.collate_fn]
    except KeyError as e:
        logger.error("collate_fn with name '%s' does not exist", config.data.collate_fn)
        raise KeyError from e

    train_loader = get_data_loader(train_dataset, batch_size=config.training.batch_size, shuffle=True,
                                   sampler=train_sampler, pin_memory=config.job.pin_memory,
                                   collate_fn=collate_fn,
                                   num_workers=num_workers)
    validation_loader = get_data_loader(validation_dataset, batch_size=config.training.batch_size,
                                        shuffle=False, sampler=validation_sampler,
                                        pin_memory=config.job.pin_memory,
                                        collate_fn=collate_fn,
                                        num_workers=num_workers)
    return train_loader, validation_loader
