from logging import getLogger
from typing import Optional, Callable
from functools import partial
import os
from pathlib import Path
import random

from omegaconf import OmegaConf
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchaudio.datasets import LIBRISPEECH

from .augmentation import get_augmentor
from source.datasets.utils import get_data_loader

logger = getLogger(__name__)


class AudioDataset(torch.utils.data.IterableDataset):
    file_types = (".wav", '.flac')

    def __init__(self, data_sets, buffer_size: int = 1000, sample_rate=16000):
        self.data_sets = data_sets
        self.sample_rate = sample_rate
        self.audio_files = []
        self._buffer = []
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.max_buffer_size = buffer_size
        self.current_buffer_size = 0
        self.find_audio_files()

    def find_audio_files(self):
        for data_set, data_set_config in self.data_sets.items():
            root_dir = data_set_config.root
            splits = data_set_config.splits
            for split in splits:
                for file_type in self.file_types:
                    self.audio_files.extend(sorted(str(p)
                                                   for p in Path(f"{root_dir}/{split}").glob("*/*/*" + file_type)))


    def add_element(self, data):
        self._buffer.append(data)
        self.current_buffer_size += 1

    def sample_from_buffer(self):
        sample = random.choice(self._buffer)
        self.current_buffer_size -= 1
        return sample

    def __iter__(self):
        for audio_file in self.audio_files:
            # Load the audio file
            waveform, sr = torchaudio.load(audio_file)
            # Add to buffer
            self.add_element((waveform, sr))

            # If buffer is full yield sample
            if self.current_buffer_size >= self.max_buffer_size:
                yield self.sample_from_buffer()

        # Yield remaining contents of buffer
        while self.current_buffer_size > 0:
            yield self.sample_from_buffer()

    def __len__(self) -> int:
        return len(self.audio_files)


def create_dataset(config):
    return AudioDataset(data_sets=config["data_sets"])


def pad_collate(batch, augmentor: Optional[Callable] = None):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data[0].T for data in batch]
    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in waveforms]  # Instance norm
    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0)
    if augmentor:
        waveforms_padded = augmentor(waveforms_padded.transpose(-1, -2))

    return waveforms_padded, lengths


def setup_dataloader(config, rank: Optional[int] = None, world_size: Optional[int] = None):
    logger.info("Creating data augmentor.")
    augmentor = get_augmentor(config)

    # Create datapipe and dataloaders
    logger.info("Creating datasets...")
    logger.info(f"Data Config: {OmegaConf.to_object(config.data)}")
    train_config = config.data.train
    validation_config = config.data.validation

    train_dataset, label_mapping = create_dataset(train_config)
    validation_dataset, _ = create_dataset(validation_config)

    logger.info("Creating data samplers...")
    if config.job.device == "distributed":
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
        val_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
    else:
        train_sampler = None
        val_sampler = None
    logger.info("Creating dataloaders...")
    train_loader = get_data_loader(train_dataset, batch_size=config.data.batch_size, shuffle=True,
                                   sampler=train_sampler, pin_memory=config.job.pin_memory,
                                   collate_fn=partial(pad_collate, augmentor=augmentor))
    validation_loader = get_data_loader(validation_dataset, batch_size=config.data.batch_size,
                                        shuffle=False, sampler=val_sampler,
                                        pin_memory=config.job.pin_memory,
                                        collate_fn=partial(pad_collate, augmentor=augmentor))
    return train_loader, validation_loader, label_mapping
