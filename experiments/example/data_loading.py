from logging import getLogger
from typing import Optional, Callable
from functools import partial
import os
import random

import pandas as pd
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, Sampler
from torch.nn.utils.rnn import pad_sequence
from torchaudio.datasets import LIBRISPEECH

from .augmentation import get_augmentor

logger = getLogger(__name__)


def read_speaker_mapping(librispeech_root_dir, split):
    """Reads the speaker mapping from a CSV file and returns a dictionary."""
    # Construct the path to the speaker mapping CSV file
    csv_file_path = os.path.join(librispeech_root_dir, split, "speaker_mapping.csv")

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Create a dictionary mapping from the DataFrame
        speaker_mapping = pd.Series(df['Class Label'].values, index=df['Speaker ID']).to_dict()

        return speaker_mapping

    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def combine_label_mappings(*mappings):
    """Combines multiple label mappings into a new mapping with unique class labels."""
    combined_mapping = {}
    current_label = 0

    for mapping in mappings:
        for speaker_id, class_label in mapping.items():
            # Check if the speaker ID is already in the combined mapping
            if speaker_id not in combined_mapping:
                combined_mapping[speaker_id] = current_label
                current_label += 1

    return combined_mapping


def create_dataset(config):
    datasets = []
    label_mappings = []
    for data_set, data_set_config in config["data_sets"].items():
        if data_set == "LibriSpeech":
            root = data_set_config.root
            splits = data_set_config.splits
            for split in splits:
                datasets.append(LIBRISPEECH(root=root, url=split, folder_in_archive=""))
                label_mappings.append(read_speaker_mapping(root, split))
        else:
            logger.error(f"Dataset {data_set} not supported")

    label_mapping = combine_label_mappings(*label_mappings)

    return torch.utils.data.ConcatDataset(datasets), label_mapping


class SpeakerTripletDataset(Dataset):
    def __init__(self, root, subset="dev-clean", transform=None, folder_in_archive=""):
        """
        Dataset that provides triplets (reference, positive, negative) for speaker verification.

        Parameters:
            root (str): Path to the root directory of the LibriSpeech dataset.
            subset (str): The subset of LibriSpeech to use ("dev-clean", "test-clean", etc.)
            transform (callable, optional): Optional transform to be applied on an audio sample.
        """
        self.dataset = LIBRISPEECH(root, url=subset, folder_in_archive=folder_in_archive)
        self.transform = transform

        # Group utterances by speaker ID
        self.speakers = {}
        for idx in range(len(self.dataset)):
            _, _, _, speaker_id, *_ = self.dataset[idx]
            if speaker_id not in self.speakers:
                self.speakers[speaker_id] = []
            self.speakers[speaker_id].append(idx)

        # Filter out speakers with only one utterance
        self.speakers = {k: v for k, v in self.speakers.items() if len(v) > 1}
        self.speaker_ids = list(self.speakers.keys())

        # Create a list of references to iterate over
        self.references = []
        for speaker_id, utterances in self.speakers.items():
            self.references.extend((speaker_id, idx) for idx in utterances)

    def __len__(self):
        return len(self.references)

    def __getitem__(self, idx):
        """
        Returns a triplet of utterances: (reference, positive, negative).

        Returns:
            ref_waveform (Tensor): Reference utterance waveform.
            pos_waveform (Tensor): Positive utterance waveform from the same speaker.
            neg_waveform (Tensor): Negative utterance waveform from a different speaker.
            sample_rate (int): Sample rate of the audio files.
        """
        speaker_id, ref_idx = self.references[idx]

        # Select a positive sample from the same speaker but different utterance
        positive_idx = random.choice([i for i in self.speakers[speaker_id] if i != ref_idx])

        # Select a negative sample from a different speaker
        negative_speaker_id = random.choice([s for s in self.speaker_ids if s != speaker_id])
        negative_idx = random.choice(self.speakers[negative_speaker_id])

        # Load the utterances
        ref_waveform, sample_rate, _, _, _, _ = self.dataset[ref_idx]
        pos_waveform, _, _, _, _, _ = self.dataset[positive_idx]
        neg_waveform, _, _, _, _, _ = self.dataset[negative_idx]

        # Apply transformations if any
        if self.transform:
            ref_waveform = self.transform(ref_waveform)
            pos_waveform = self.transform(pos_waveform)
            neg_waveform = self.transform(neg_waveform)

        return ref_waveform, pos_waveform, neg_waveform, sample_rate


def pad_collate(batch, augmentor: Callable = lambda x: x):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data[0].T for data in batch]
    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in waveforms]  # Instance norm
    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])
    speaker_id = torch.tensor([data[3] for data in batch])

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0).transpose(-1, -2)
    waveforms_padded = augmentor(waveforms_padded)

    return waveforms_padded, lengths, speaker_id


def pad_collate_eval(batch, augmentor: Callable = lambda x: x):
    """Padding function used to deal with batches of sequences of variable lengths."""
    ref_waveforms = [data[0].T for data in batch]
    ref_waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in ref_waveforms]  # Instance norm
    ref_lengths = torch.tensor([waveform.size(0) for waveform in ref_waveforms])
    ref_waveforms_padded = pad_sequence(ref_waveforms, batch_first=True, padding_value=0)
    ref_waveforms_padded = augmentor(ref_waveforms_padded.transpose(-1, -2))

    pos_waveforms = [data[1].T for data in batch]
    pos_waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in
                     pos_waveforms]  # Instance norm
    pos_lengths = torch.tensor([waveform.size(0) for waveform in pos_waveforms])
    pos_waveforms_padded = pad_sequence(pos_waveforms, batch_first=True, padding_value=0)
    pos_waveforms_padded = augmentor(pos_waveforms_padded.transpose(-1, -2))

    neg_waveforms = [data[2].T for data in batch]
    neg_waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in
                     neg_waveforms]  # Instance norm
    neg_lengths = torch.tensor([waveform.size(0) for waveform in neg_waveforms])
    neg_waveforms_padded = pad_sequence(neg_waveforms, batch_first=True, padding_value=0)
    neg_waveforms_padded = augmentor(neg_waveforms_padded.transpose(-1, -2))

    return ref_waveforms_padded, ref_lengths, pos_waveforms_padded, pos_lengths, neg_waveforms_padded, neg_lengths


def create_evaluation_dataset(config):
    datasets = []
    label_mappings = []
    for data_set, data_set_config in config["data_sets"].items():
        if data_set == "LibriSpeech":
            root = data_set_config.root
            splits = data_set_config.splits
            for split in splits:
                datasets.append(SpeakerTripletDataset(root=root, subset=split, folder_in_archive=""))
                label_mappings.append(read_speaker_mapping(root, split))
        else:
            logger.error(f"Dataset {data_set} not supported")

    label_mapping = combine_label_mappings(*label_mappings)

    return torch.utils.data.ConcatDataset(datasets), label_mapping


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=pad_collate,
                    pin_memory: bool = False, sampler: Optional[Sampler] = None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, sampler=sampler)
    return data_loader


def setup_dataloader(config, distributed=False):
    logger.info("Creating data augmentor.")
    augmentor = get_augmentor(config)

    # Create datapipe and dataloaders
    logger.info("Creating datasets...")
    logger.info(f"Data Config: {OmegaConf.to_object(config.data)}")
    train_config = config.data.train
    validation_config = config.data.validation

    train_dataset, label_mapping = create_dataset(train_config)
    validation_dataset, _ = create_evaluation_dataset(validation_config)

    logger.info("Creating data samplers...")
    if distributed:
        # Distributed Sampler: Ensures data is divided among GPUs using DistributedSampler.
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
        val_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
    else:
        train_sampler = None
        val_sampler = None

    logger.info("Creating dataloaders...")
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"] if distributed else config.job.num_workers)
    train_loader = get_data_loader(train_dataset, batch_size=config.training.batch_size, shuffle=True,
                                   sampler=train_sampler, pin_memory=config.job.pin_memory,
                                   collate_fn=partial(pad_collate, augmentor=augmentor),
                                   num_workers=num_workers)
    validation_loader = get_data_loader(validation_dataset, batch_size=config.training.batch_size,
                                        shuffle=False, sampler=val_sampler,
                                        pin_memory=config.job.pin_memory,
                                        collate_fn=partial(pad_collate_eval, augmentor=augmentor),
                                        num_workers=num_workers)
    return train_loader, validation_loader, label_mapping



