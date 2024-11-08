from pathlib import Path
from functools import partial
from typing import Optional, Tuple, Callable, Union
from io import BytesIO
import random
from math import ceil

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchdata import datapipes as dp

from common.misc import count_files

implemented_data_sets = ["librispeech_concat"]

librispeech_splits = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360",
                      "train-other-500"]


def filter_fn(filename, file_types):
    return filename.endswith(file_types)


def decode(item):
    key, value = item

    text_file_ending = (".txt", ".bin", ".sr", ".hop_length", ".window_length", ".n_fft", ".n_mels")

    if key.endswith(text_file_ending):
        return key, value.read().decode("utf-8")
    else:
        return key, value.read()

def is_audio_path(item):
    a = item.endswith(".wav") or item.endswith(".flac")
    b = not "_" in item.split(".")[-2]
    return a and b  # and not "_" in item.split(".")[-2]


def to_item(wds_item, load_from: str):
    if load_from == "raw":
        if ".flac" in wds_item.keys():
            audio_ext = ".flac"
        else:
            audio_ext = ".wav"
        return wds_item["__key__"], wds_item[audio_ext]
    elif load_from == "decoded":
        if load_from == "raw":
            return wds_item["__key__"], wds_item[".pth"], wds_item[".sr"]
    return wds_item["__key__"], wds_item[".pth"]

def load_raw_waveform(item):
    audio_path, file_stream = item
    with BytesIO(file_stream) as fd:
        waveform, sampling_rate = torchaudio.load(fd)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1.e-9)
    return waveform, sampling_rate, audio_path


def load_decoded_waveform(item):
    waveform_path, file_stream, sampling_rate = item

    with BytesIO(file_stream) as fd:
        waveform = torch.load(fd)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1.e-9)
    return waveform, sampling_rate, waveform_path


def load_features(item):
    feature_path, file_stream = item

    with BytesIO(file_stream) as fd:
        features = torch.load(fd)
    features.requires_grad = True
    return features, feature_path


def augment(item, augmentor):
    waveform, sampling_rate, audio_path = item
    return augmentor(waveform.unsqueeze(0))[0], sampling_rate, audio_path


def extract_features(item, feature_extractor):
    waveform, sampling_rate, audio_path = item
    return feature_extractor(waveform)


def segment_features(item, segment_size, randomize_start: bool = False, drop_last: bool = False,
                     min_length: int = 0):
    features = item
    length = features.size(-1)
    n_segments = length // segment_size
    if randomize_start:
        remainder = length % segment_size
        start = random.randint(0, remainder)
    else:
        start = 0
    features = features[:, :, start:].split(segment_size, dim=-1)
    if (drop_last or (len(features[-1]) < min_length) and (len(features) > 1)):
        features = features[:-1]
    return features


def pad_collate_features(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    features = [data[0].T for data in batch[0]]
    features = [(feature - feature.mean()) / (feature.std() + 1.e-9) for feature in features]  # Instance norm

    lengths = torch.tensor([feature.size(0) for feature in features])

    features_padded = pad_sequence(features, batch_first=True, padding_value=0)

    return features_padded, lengths


def pad_collate_features_clean_noisy(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    features = [data[0][0].T for data in batch[0]]
    noisy_features = [data[1][0].T for data in batch[0]]

    features = [(feature - feature.mean()) / (feature.std() + 1.e-9)
                for feature in features]  # Instance norm
    noisy_features = [(feature - feature.mean()) / (feature.std() + 1.e-9)
                      for feature in noisy_features]  # Instance norm

    lengths = torch.tensor([feature.size(0) for feature in features])

    targets_padded = pad_sequence(features, batch_first=True, padding_value=0)
    features_padded = pad_sequence(noisy_features, batch_first=True, padding_value=0)

    return features_padded, targets_padded, lengths


def build_apc_datapipe(data_sets, feature_extractor,
                       augmentor: Callable = None,
                       load_from_tar: bool = False,
                       buffer_size: int = 10000,
                       batch_size: int = 1,
                       max_token_count: int = 0,
                       load_from: str = "raw",
                       clean_and_noisy: bool = False,
                       segment_max_size=None,
                       min_length: int = 0):
    if load_from in ["decoded", "features"]:
        file_types = (".pth", ".pt")
    elif load_from == "raw":
        file_types = (".wav", '.flac')
    else:
        assert ValueError(f"Loading_method: {load_from} not supported. "
                          "Supported methods are 'raw', 'decoded' and 'features'")

    # List all files
    datapipes = []
    for data_set_name, info in data_sets.items():
        for split in info["splits"]:
            if load_from_tar:
                datapipe = dp.iter.FileLister(info["root"] + split, "*.tar")
            else:
                datapipe = dp.iter.FileLister(info["root"] + split, recursive=True,
                                              masks=[f"*{file_type}" for file_type in file_types])
            datapipes.append(datapipe)

    # Concatenate file lists
    datapipe = dp.iter.Concater(*datapipes)

    # Shuffle files and apply sharding filter
    datapipe = datapipe.shuffle(buffer_size=buffer_size).sharding_filter()

    # Open files
    datapipe = dp.iter.FileOpener(datapipe, mode="b")
    if load_from_tar:
        datapipe = datapipe.load_from_tar().map(decode).webdataset().map(partial(to_item, load_from=load_from))
    else:
        datapipe = datapipe.map(decode)

    # Load features or generate from waveforms
    if load_from == "features":
        datapipe = datapipe.map(load_features)
    else:
        if load_from == "decoded":
            datapipe = datapipe.map(load_decoded_waveform)
        else:
            datapipe = datapipe.map(load_raw_waveform)
        if augmentor:
            if clean_and_noisy:
                datapipe, datapipe_augmented = datapipe.fork(num_instances=2)
                datapipe_augmented = datapipe_augmented.map(partial(augment, augmentor=augmentor))
                datapipe = datapipe.map(partial(extract_features, feature_extractor=feature_extractor))
                datapipe_augmented = datapipe_augmented.map(partial(extract_features, feature_extractor=feature_extractor))
                if segment_max_size:
                    datapipe = datapipe.flatmap(
                        partial(segment_features, segment_size=segment_max_size, 
                                min_length=min_length, drop_last=False, randomize_start=False))
                    datapipe_augmented = datapipe_augmented.flatmap(
                        partial(segment_features, segment_size=segment_max_size,
                                min_length=min_length, drop_last=False, randomize_start=False))
                    datapipe = datapipe.zip(datapipe_augmented)
                    datapipe = datapipe.shuffle(buffer_size=1000)
                else:
                    datapipe = datapipe.zip(datapipe_augmented)
            else:
                datapipe = datapipe.map(partial(augment, augmentor=augmentor))
                datapipe = datapipe.map(partial(extract_features, feature_extractor=feature_extractor))
                if segment_max_size:
                    datapipe = datapipe.flatmap(
                        partial(segment_features, segment_size=segment_max_size, 
                                min_length=min_length, drop_last=False, randomize_start=False))
                    datapipe = datapipe.shuffle(buffer_size=1000)
        else:
            datapipe = datapipe.map(partial(extract_features, feature_extractor=feature_extractor))
            if segment_max_size:
                    datapipe = datapipe.flatmap(
                        partial(segment_features, segment_size=segment_max_size, 
                                min_length=min_length, drop_last=False, randomize_start=False))
                    datapipe = datapipe.shuffle(buffer_size=1000)

    
    if max_token_count:
        if clean_and_noisy:
            def len_fn(features):
                return features[0].size(-1)
        else:
            def len_fn(features):
                return features.size(-1)
        datapipe = datapipe.max_token_bucketize(max_token_count=max_token_count, len_fn=len_fn, include_padding=True,
                                                buffer_size=100, min_len=min_length)
    else:
        datapipe = datapipe.batch(batch_size=batch_size, drop_last=True)
    datapipe = datapipe.set_length(1)
    return datapipe


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=pad_collate_features,
                    pin_memory: bool = False):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    return data_loader


if __name__ == "__main__":

    from torchaudio.transforms import MelSpectrogram

    import numpy as np
    from librosa.feature import melspectrogram

    from torch import nn
    from torch.nn import functional as F


    class FeatureExtractor:
        def __init__(self, window_length=400, hop_length=160, n_fft=400, n_mels=40, sample_rate=16000,
                     stacked_consecutive_features: int = 1, subsample_factor: int = 1):
            self.window_length = window_length
            self.hop_length = hop_length
            self.n_fft = n_fft
            self.n_mels = n_mels
            self.sample_rate = sample_rate
            self.mel_spectrogram = MelSpectrogram(sample_rate=self.sample_rate,
                                                  n_fft=self.n_fft,
                                                  n_mels=self.n_mels,
                                                  win_length=self.window_length,
                                                  hop_length=self.hop_length
                                                  )
            self.stacked_consecutive_features = stacked_consecutive_features
            self.subsample_factor = subsample_factor
            self.subsample = nn.AvgPool1d(subsample_factor, stride=subsample_factor)
            self.feature_rate = sample_rate / (hop_length * subsample_factor)

        def __call__(self, waveform: torch.Tensor):
            # mel_features = self.mel_spectrogram(waveform)
            # log_mel_features = power_to_db(mel_features)
            mel_features = melspectrogram(y=waveform.numpy(), sr=self.sample_rate, n_fft=self.n_fft,
                                          hop_length=self.hop_length, n_mels=self.n_mels)
            # log_mel_features = torch.log10(mel_features + 1e-6)
            log_mel_features = torch.from_numpy(np.log10(mel_features + 1.e-6))

            if self.stacked_consecutive_features > 1:
                log_mel_features = self.stack_features(log_mel_features)
            if self.subsample_factor > 1:
                log_mel_features = self.subsample(log_mel_features)
            return log_mel_features

        def stack_features(self, features: torch.Tensor):
            # check if padding is needed after stacking features
            if features.size(-1) % self.stacked_consecutive_features:
                padding = self.stacked_consecutive_features - \
                          (features.size(-1) % self.stacked_consecutive_features)

                features = F.pad(features, (0, padding))
            features = features.permute(0, 2, 1)
            features = features.reshape(features.size(0), -1,
                                        features.size(2) * self.stacked_consecutive_features)
            features = features.permute(0, 2, 1)
            return features


    feature_extractor = FeatureExtractor()

    librispeech_root = "/Users/JG96XG/Desktop/data_sets/LibriSpeech/sharded/raw/" #"/Users/holge/Downloads/LibriSpeech/"
    librispeech_splits = ["train-clean-100"]
    # voxceleb1_root = "/Users/JG96XG/Desktop/data_sets/VoxCeleb/voxceleb_1/"
    # voxceleb1_splits = ["test"]
    # voxceleb2_root = "/Users/JG96XG/Desktop/data_sets/VoxCeleb/voxceleb_2/"
    # voxceleb2_splits = ["test"]

    data_sets = {"librispeech": {"root": librispeech_root, "splits": librispeech_splits}}


    def augmentor(waveform):
        return waveform ** 2


    datapipe = build_apc_datapipe(data_sets, feature_extractor=feature_extractor,
                                  buffer_size=1000, load_from="raw", load_from_tar=True,
                                  clean_and_noisy=True, augmentor=augmentor, max_token_count=3000,
                                  segment_max_size=500,
                                  min_length=200)

    #it1 = iter(datapipe)
    dataloader = get_data_loader(datapipe, collate_fn=pad_collate_features_clean_noisy, batch_size=1)
    for data in dataloader:
        feats, feats_noisy, lengths = data
    print("done")
