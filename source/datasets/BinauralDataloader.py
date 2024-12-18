from functools import partial
from typing import Callable, Optional
from io import BytesIO
import os
from operator import itemgetter
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchaudio.functional import resample
from torchdata import datapipes as dp
from torch.utils.data import DataLoader, Sampler

from source.simulators import BinauralAudioSimulator
from source.utils import spherical_to_cartesian, cartesian_to_spherical


def filter_fn(filename, file_types):
    return filename.endswith(file_types)


def decode(item):
    key, value = item

    text_file_ending = (".txt", ".bin", ".sr", ".hop_length", ".window_length", ".n_fft", ".n_mels")

    if key.endswith(text_file_ending):
        return key, value.read().decode("utf-8")
    return key, value.read()


def is_audio_path(item):
    a = item.endswith(".wav") or item.endswith(".flac")
    b = "_" not in item.split(".")[-2]
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


def resample_waveform(item, desired_sampling_rate):
    waveform, sampling_rate, waveform_path = item
    if sampling_rate != desired_sampling_rate:
        waveform = resample(waveform, sampling_rate, desired_sampling_rate)
    return waveform, sampling_rate, waveform_path


def to_binaural(item, hrtf_simulator):
    waveform, sampling_rate, audio_path = item
    elevation = random.randrange(0, 120, 10)
    azimuth = random.randrange(0, 360, 5)
    head_direction = (1.5, elevation, azimuth)  # Left, horizontal plane
    head_direction = spherical_to_cartesian(*head_direction)
    left_signal, right_signal, receiver_position = hrtf_simulator(waveform.view(1, 1, -1),
                                                                  source_direction=head_direction)
    stereo = torch.cat((left_signal, right_signal), dim=0).squeeze()
    stereo = stereo / (torch.max(torch.abs(stereo)) + 1.e-9)

    return stereo, sampling_rate, audio_path, head_direction, receiver_position


def augment(item, augmentor):
    waveform, sampling_rate, audio_path, head_direction, receiver_position = item
    return augmentor(waveform.unsqueeze(0))[0], sampling_rate, audio_path, head_direction, receiver_position


def segment(item, segment_size, randomize_start: bool = False, drop_last: bool = False,
            min_length: int = 0):
    source, sampling_rate, audio_path, head_direction, receiver_position = item
    length = source.size(-1)
    if randomize_start:
        remainder = length % segment_size
        start = random.randint(0, remainder)
    else:
        start = 0
    source = source[:, start:].split(segment_size, dim=-1)
    if drop_last or (len(source[-1]) < min_length) and (len(source) > 1):
        source = source[:-1]

    out = [(audio_segment, sampling_rate, audio_path, head_direction, receiver_position) for audio_segment in source]

    return out


def pad_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data[0].transpose(-1, -2) for data in batch[0]]
    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in waveforms]  # Instance norm
    head_direction = [data[3] for data in batch[0]]
    receiver_position = [data[4] for data in batch[0]]

    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0)

    return waveforms_padded, lengths, head_direction, receiver_position


def pad_collate_clean_noisy(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data[0].transpose(-1, -2) for data in batch[0]]
    noisy_waveforms = [data[1].transpose(-1, -2) for data in batch[0]]

    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9)
                 for waveform in waveforms]  # Instance norm
    noisy_waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9)
                       for waveform in noisy_waveforms]  # Instance norm

    head_direction = [data[4] for data in batch[0]]
    receiver_position = [data[5] for data in batch[0]]

    lengths = torch.tensor([waveform.size(-1) for waveform in waveforms])

    clean_waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0).transpose(-1, -2)
    noisy_waveforms_padded = pad_sequence(noisy_waveforms, batch_first=True, padding_value=0).transpose(-1, -2)

    return clean_waveforms_padded, noisy_waveforms_padded, lengths, head_direction, receiver_position


def build_binaural_datapipe(data_sets,
                            hrtf_folder,
                            sampling_rate: int = 16000,
                            augmentor: Optional[Callable] = None,
                            load_from_tar: bool = False,
                            buffer_size: int = 10000,
                            batch_size: int = 1,
                            max_token_count: int = 0,
                            load_from: str = "raw",
                            clean_and_noisy: bool = False,
                            segment_max_size=None,
                            min_length: int = 0):
    if load_from == "decoded":
        file_types = (".pth", ".pt")
    elif load_from == "raw":
        file_types = (".wav", '.flac')
    else:
        raise ValueError(f"Loading_method: {load_from} not supported. "
                          "Supported methods are 'raw' and 'decoded'")
    # Create hrtf simulator
    hrtf_simulator = BinauralAudioSimulator(hrtf_folder=hrtf_folder, sampling_rate=sampling_rate)

    # List all files
    datapipes = []
    for data_set_name, info in data_sets.items():
        for split in info["splits"]:
            if load_from_tar:
                datapipe = dp.iter.FileLister(os.path.join(info["root"], split), "*.tar")
            else:
                datapipe = dp.iter.FileLister(os.path.join(info["root"], split), recursive=True,
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

    # Load waveforms
    if load_from == "decoded":
        datapipe = datapipe.map(load_decoded_waveform)
    else:
        datapipe = datapipe.map(load_raw_waveform)

    # Resample waveforms if necesarry
    datapipe = datapipe.map(partial(resample_waveform, desired_sampling_rate=sampling_rate))

    # Augment if necessary
    if augmentor:
        # Split if both clean and noisy version of the signal is needed
        if clean_and_noisy:
            def merge_fn(t1, t2):
                return t1[0], t2[0], t1[1], t1[2], t1[3], t1[4]

            datapipe = datapipe.map(partial(to_binaural, hrtf_simulator=hrtf_simulator))
            datapipe, datapipe_augmented = datapipe.fork(num_instances=2)
            datapipe_augmented = datapipe_augmented.map(partial(augment, augmentor=augmentor))
            if segment_max_size:
                datapipe = datapipe.flatmap(
                    partial(segment, segment_size=segment_max_size,
                            min_length=min_length, drop_last=False, randomize_start=False))
                datapipe_augmented = datapipe_augmented.flatmap(
                    partial(segment, segment_size=segment_max_size,
                            min_length=min_length, drop_last=False, randomize_start=False))
                datapipe = datapipe.zip_with_iter(datapipe_augmented,
                                                  key_fn=itemgetter(2),
                                                  ref_key_fn=itemgetter(2),
                                                  merge_fn=merge_fn)
                datapipe = datapipe.shuffle(buffer_size=1000)
            else:
                datapipe = datapipe.zip_with_iter(datapipe_augmented,
                                                  key_fn=itemgetter(2),
                                                  ref_key_fn=itemgetter(2),
                                                  merge_fn=merge_fn)
        else:

            datapipe = datapipe.map(partial(to_binaural, hrtf_simulator=hrtf_simulator))
            datapipe = datapipe.map(partial(augment, augmentor=augmentor))
            if segment_max_size:
                datapipe = datapipe.flatmap(
                    partial(segment, segment_size=segment_max_size,
                            min_length=min_length, drop_last=False, randomize_start=False))
                datapipe = datapipe.shuffle(buffer_size=1000)
    else:
        datapipe = datapipe.map(partial(to_binaural, hrtf_simulator=hrtf_simulator))
        if segment_max_size:
            datapipe = datapipe.flatmap(
                partial(segment, segment_size=segment_max_size,
                        min_length=min_length, drop_last=False, randomize_start=False))
            datapipe = datapipe.shuffle(buffer_size=1000)

    # Batch either using max bucket batching or standard random sampling
    if max_token_count:
        def len_fn(source):
            return source[0].size(-1)

        datapipe = datapipe.max_token_bucketize(max_token_count=max_token_count, len_fn=len_fn, include_padding=True,
                                                buffer_size=100, min_len=min_length)
    else:
        datapipe = datapipe.batch(batch_size=batch_size, drop_last=True)
    return datapipe


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=pad_collate_clean_noisy,
                    pin_memory: bool = False, sampler: Optional[Sampler] = None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
                             sampler=sampler)
    return data_loader


if __name__ == "__main__":
    import time
    import sounddevice as sd
    from torch_audiomentations import AddBackgroundNoise

    librispeech_root = "/Users/JG96XG/Desktop/data_sets/LibriSpeech/sharded/raw/"  #"/Users/holge/Downloads/LibriSpeech/"
    librispeech_splits = ["train-clean-100"]
    data_sets = {"librispeech": {"root": librispeech_root, "splits": librispeech_splits}}

    sampling_rate = 16000

    noise_files_path = "/Users/JG96XG/Desktop/data_sets/noise_files/kolbaek_noise_files/bus/"
    noise_augmenter = AddBackgroundNoise(background_paths=noise_files_path, p=1.0,
                                         min_snr_in_db=10, max_snr_in_db=10,
                                         sample_rate=sampling_rate)


    datapipe = build_binaural_datapipe(data_sets, hrtf_folder="/Users/JG96XG/Desktop/data_sets/HRTFs/RIEC_hrtf_all/",
                                       buffer_size=10, load_from="raw", load_from_tar=True,
                                       sampling_rate=sampling_rate,
                                       clean_and_noisy=True, augmentor=noise_augmenter, max_token_count=96000,
                                       segment_max_size=48000,
                                       min_length=200)

    dataloader = get_data_loader(datapipe, collate_fn=pad_collate_clean_noisy, batch_size=1)
    for data in dataloader:
        wavs, wavs_noisy, lengths, head_direction, receiver_position = data
        stereo = wavs[0].numpy().astype(np.float64)
        direction = cartesian_to_spherical(*head_direction[0])
        sd.play(stereo.T*0.1, 16000)
        time.sleep(3.0)
        sd.stop()
        stereo = wavs_noisy[0].numpy().astype(np.float64)
        direction = cartesian_to_spherical(*head_direction[0])
        sd.play(stereo.T * 0.1, 16000)
        time.sleep(3.0)
        sd.stop()

    print("done")
