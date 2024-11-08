from functools import partial
from typing import Optional, Callable
from io import BytesIO
import random
from operator import itemgetter

import torch
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchdata import datapipes as dp


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


def augment(item, augmentor):
    waveform, sampling_rate, audio_path = item
    return augmentor(waveform.unsqueeze(0))[0], sampling_rate, audio_path


def segment(item, segment_size, randomize_start: bool = False, drop_last: bool = False,
                     min_length: int = 0):
    source, sampling_rate, audio_path = item
    length = source.size(-1)
    if randomize_start:
        remainder = length % segment_size
        start = random.randint(0, remainder)
    else:
        start = 0
    source = source[:, start:].split(segment_size, dim=-1)
    if (drop_last or (len(source[-1]) < min_length) and (len(source) > 1)):
        source = source[:-1]

    out = [(audio_segment, sampling_rate, audio_path) for audio_segment in source]

    return out


def pad_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data[0].T for data in batch[0]]
    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in waveforms]  # Instance norm

    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0)

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

    return noisy_waveforms_padded, clean_waveforms_padded, lengths


def build_bpc_datapipe(data_sets,
                       augmentor: Callable = None,
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
        assert ValueError(f"Loading_method: {load_from} not supported. "
                          "Supported methods are 'raw' and 'decoded'")

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

    # Load waveforms

    if load_from == "decoded":
        datapipe = datapipe.map(load_decoded_waveform)
    else:
        datapipe = datapipe.map(load_raw_waveform)
    if augmentor:
        if clean_and_noisy:
            def merge_fn(t1, t2):
                return (t1[0], t2[0], t1[1], t1[2])
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
            datapipe = datapipe.map(partial(augment, augmentor=augmentor))
            if segment_max_size:
                datapipe = datapipe.flatmap(
                    partial(segment, segment_size=segment_max_size,
                            min_length=min_length, drop_last=False, randomize_start=False))
                datapipe = datapipe.shuffle(buffer_size=1000)
    else:
        if segment_max_size:
                datapipe = datapipe.flatmap(
                    partial(segment, segment_size=segment_max_size,
                            min_length=min_length, drop_last=False, randomize_start=False))
                datapipe = datapipe.shuffle(buffer_size=1000)
    
    if max_token_count:
        def len_fn(source):
            return source[0].size(-1)

        datapipe = datapipe.max_token_bucketize(max_token_count=max_token_count, len_fn=len_fn, include_padding=True,
                                                buffer_size=100, min_len=min_length)
    else:
        datapipe = datapipe.batch(batch_size=batch_size, drop_last=True)
    datapipe = datapipe.set_length(1)
    return datapipe


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=pad_collate_clean_noisy,
                    pin_memory: bool = False, sampler: Optional[Sampler] = None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
                             sampler=sampler)
    return data_loader


if __name__ == "__main__":
    librispeech_root = "/Users/JG96XG/Desktop/data_sets/LibriSpeech/sharded/raw/" #"/Users/holge/Downloads/LibriSpeech/"
    librispeech_splits = ["train-clean-100"]
    # voxceleb1_root = "/Users/JG96XG/Desktop/data_sets/VoxCeleb/voxceleb_1/"
    # voxceleb1_splits = ["test"]
    # voxceleb2_root = "/Users/JG96XG/Desktop/data_sets/VoxCeleb/voxceleb_2/"
    # voxceleb2_splits = ["test"]

    data_sets = {"librispeech": {"root": librispeech_root, "splits": librispeech_splits}}


    def augmentor(waveform):
        return waveform ** 2


    datapipe = build_bpc_datapipe(data_sets,
                                  buffer_size=1000, load_from="raw", load_from_tar=True,
                                  clean_and_noisy=True, augmentor=augmentor, max_token_count=3000,
                                  segment_max_size=500,
                                  min_length=200)

    #it1 = iter(datapipe)
    dataloader = get_data_loader(datapipe, collate_fn=pad_collate_clean_noisy, batch_size=1)
    for data in dataloader:
        feats, feats_noisy, lengths = data
    print("done")
