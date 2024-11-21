from logging import getLogger
from typing import Optional, Callable
from functools import partial
import os

from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
import torch
from torch.nn.utils.rnn import pad_sequence

from source.datasets import BinauralLibriSpeechDataset, get_data_loader
from .augmentation import get_augmentor

logger = getLogger(__name__)


def generate_data_set(config):
    datasets = []
    for data_set, data_set_config in config["data_sets"].items():
        if data_set == "BinauralLibriSpeech":
            root = str(os.path.join(data_set_config.root, data_set_config.subset))
            splits = data_set_config.splits
            metadata_filename = data_set_config.metadata_filename
            for split in splits:
                dataset = BinauralLibriSpeechDataset(root_dir=root,
                                                     metadata_filename=metadata_filename,
                                                     split=split)
                datasets.append(dataset)
        else:
            logger.error(f"Dataset {data_set} not supported")
    return ConcatDataset(datasets)


def pad_collate(batch, augmentor: Optional[Callable] = lambda x: x):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data["waveform"].T for data in batch]
    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in waveforms]  # Instance norm
    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])
    doas = torch.tensor([[data["elevation"], data["azimuth"]] for i, data in enumerate(batch)])

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=-1).transpose(-1, -2)
    waveforms_padded = augmentor(waveforms_padded)

    return waveforms_padded, lengths, doas


def setup_dataloader(config, distributed=False):
    logger.info("Creating data augmentor.")
    augmentor = get_augmentor(config)

    # Create datapipe and dataloaders
    logger.info("Creating datasets...")
    logger.info(f"Data Config: {OmegaConf.to_object(config.data)}")
    train_config = config.data.train
    validation_config = config.data.validation

    train_dataset= generate_data_set(train_config)
    validation_dataset = generate_data_set(validation_config)

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
    train_loader = get_data_loader(train_dataset, batch_size=config.data.train.batch_size, shuffle=True,
                                   sampler=train_sampler, pin_memory=config.job.pin_memory,
                                   collate_fn=partial(pad_collate, augmentor=augmentor),
                                   num_workers=num_workers)
    validation_loader = get_data_loader(validation_dataset, batch_size=config.data.validation.batch_size,
                                        shuffle=False, sampler=validation_sampler,
                                        pin_memory=config.job.pin_memory,
                                        collate_fn=partial(pad_collate, augmentor=augmentor),
                                        num_workers=num_workers)
    return train_loader, validation_loader
