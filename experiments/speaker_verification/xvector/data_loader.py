from pathlib import Path
from functools import partial
from typing import Optional, Tuple, Callable, Union
from io import BytesIO
import random
from math import ceil

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchdata import datapipes as dp

from common.misc import count_files, nearest_interp

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


def to_item(wds_item, load_from: str):
    if ".flac" in wds_item.keys():
        audio_ext = ".flac"
    else:
        audio_ext = ".wav"

    if load_from == "raw":
        return wds_item["__key__"], wds_item[audio_ext]
    elif load_from == "decoded":
        if load_from == "raw":
            return wds_item["__key__"], wds_item[".pth"], wds_item[".sr"]
    return wds_item["__key__"], wds_item[".pth"]


def load_raw_waveform(item):
    audio_path, file_stream = item
    if "LibriSpeech" in audio_path:
        speaker_id, chapter_id, utterance_id = str(Path(audio_path).stem).split("-")
    elif "VoxCeleb" in audio_path:
        speaker_id, chapter_id, utterance_id = str(audio_path.strip(".wav")).split("/")[-3:]
    else:
        speaker_id, chapter_id, utterance_id = -1, -1, -1
        assert ValueError("Data path does not include LibriSpeech or VoxCeleb")

    with BytesIO(file_stream) as fd:
        waveform, sampling_rate = torchaudio.load(fd)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1.e-9)
    return waveform, sampling_rate, speaker_id, audio_path


def load_decoded_waveform(item):
    waveform_path, file_stream, sampling_rate = item
    if "LibriSpeech" in waveform_path:
        speaker_id, chapter_id, utterance_id = str(Path(waveform_path).stem).split("-")
    elif "VoxCeleb" in waveform_path:
        speaker_id, chapter_id, utterance_id = str(waveform_path).split("/")[-3:]
    else:
        speaker_id, chapter_id, utterance_id = -1, -1, -1
        print("Data path does not include LibriSpeech or VoxCeleb")

    with BytesIO(file_stream) as fd:
        waveform = torch.load(fd)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1.e-9)
    return waveform, sampling_rate, speaker_id, waveform_path


def load_features(item):
    feature_path, file_stream = item
    if "LibriSpeech" in feature_path:
        speaker_id, chapter_id, utterance_id = str(Path(feature_path).stem).split("-")
    elif "VoxCeleb" in feature_path:
        speaker_id, chapter_id, utterance_id = str(feature_path).split("/")[-3:]
    else:
        speaker_id, chapter_id, utterance_id = -1, -1, -1
        print("Data path does not include LibriSpeech or VoxCeleb")

    with BytesIO(file_stream) as fd:
        features = torch.load(fd)
    features.requires_grad = True
    return features, speaker_id, feature_path


def remove_silence(item, vad):
    waveform, sampling_rate, speaker_id, waveform_path = item
    vad_label, vad_timestamp = vad(waveform[0].numpy(), sampling_rate=sampling_rate)
    vad_label_interpolated = nearest_interp(torch.arange(waveform.size(-1)) / sampling_rate,
                                            torch.from_numpy(vad_timestamp),
                                            torch.from_numpy(vad_label))
    waveform_vad = waveform[:, vad_label_interpolated.bool()]
    return waveform_vad, sampling_rate, speaker_id, waveform_path


def augment(item, augmentor):
    waveform, sampling_rate, speaker_id, audio_path = item
    return augmentor(waveform.unsqueeze(0))[0], sampling_rate, speaker_id, audio_path


def extract_features(item, feature_extractor):
    waveform, sampling_rate, speaker_id, audio_path = item
    return feature_extractor(waveform), speaker_id, audio_path


def segment_features(item, segment_size,
                     randomize_start: bool = False,
                     drop_last: bool = False,
                     min_length: int = 0):
    features, speaker_id, audio_path = item
    length = features.size(-1)
    n_segments = length // segment_size
    if randomize_start:
        remainder = length % segment_size
        start = random.randint(0, remainder)
    else:
        start = 0
    features = features[:, :, start:].split(segment_size, dim=-1)
    if (drop_last or (len(features[-1]) < min_length)) and (len(features) > n_segments):
        features = features[:-1]

    output = [(feature, speaker_id, audio_path, segment_id) for segment_id, feature in enumerate(features)]
    return output


def convert_speaker_id_to_label(item, speaker_id_to_label):
    waveform, speaker_id, audio_path = item
    return waveform, speaker_id_to_label[speaker_id], audio_path


def pad_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    features = [data[0][0].T for data in batch[0]]
    features = [(feature - feature.mean()) / (feature.std() + 1.e-9)
                for feature in features]  # Instance norm
    labels = torch.tensor([data[1] for data in batch[0]], dtype=torch.int64)
    lengths = torch.tensor([feature.size(0) for feature in features])

    features_padded = pad_sequence(features, batch_first=True, padding_value=0)

    return features_padded, lengths, labels


def build_xvector_datapipe(data_sets, feature_extractor,
                           waveforms_dir: str = "Waveforms/",
                           augmentor: Callable = None,
                           load_from_tar: bool = False,
                           buffer_size: int = 10000,
                           batch_size: int = 1,
                           max_token_count: int = 0,
                           load_from: str = "raw",
                           segment_max_size=None,
                           min_length: int = 0,
                           metadata_filename: str = "metadata.csv"):
    if load_from in ["decoded", "features"]:
        file_types = (".pth", ".pt")
    elif load_from == "raw":
        file_types = (".wav", '.flac')
    else:
        file_types = ()
        assert ValueError(f"Loading_method: {load_from} not supported. "
                          "Supported methods are 'raw', 'decoded' and 'features'")

    length = 0

    # List all files
    datapipes = []
    metadata = []
    for data_set_name, info in data_sets.items():
        for split in info["splits"]:
            if load_from_tar:
                datapipe = dp.iter.FileLister(info["root"] + split, "*.tar")
            else:
                datapipe = dp.iter.FileLister(info["root"] + split, recursive=True,
                                              masks=[f"*{file_type}" for file_type in file_types])
                length += count_files(info["root"] + split + "/" + waveforms_dir, ".flac")
            datapipes.append(datapipe)
            metadata.append(pd.read_csv(info["root"] + split + "/" + metadata_filename))
    metadata = pd.concat(metadata)
    speaker_ids = np.sort(metadata.speaker_id.unique())

    speaker_id_to_label = {str(speaker_id): i for i, speaker_id in enumerate(speaker_ids)}
    label_to_speaker_id = dict(enumerate(speaker_ids))

    # Concatenate filelists
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
            datapipe = datapipe.map(partial(augment, augmentor=augmentor))
        datapipe = datapipe.map(partial(extract_features, feature_extractor=feature_extractor))

    # Map speaker id to label
    datapipe = datapipe.map(partial(convert_speaker_id_to_label,
                                    speaker_id_to_label=speaker_id_to_label))

    # Segment if necessary
    if segment_max_size:
        datapipe = datapipe.flatmap(
            partial(segment_features, segment_size=segment_max_size,
                    min_length=min_length, drop_last=False, randomize_start=False))
        # Shuffle after segmentation
        datapipe = datapipe.shuffle(buffer_size=1000)

    # Batching
    if max_token_count:
        def len_fn(features):
            return features[0].size(-1)

        datapipe = datapipe.max_token_bucketize(max_token_count=max_token_count, len_fn=len_fn, include_padding=True,
                                                buffer_size=100, min_len=min_length)
    else:
        datapipe = datapipe.batch(batch_size=batch_size, drop_last=True)
    datapipe = datapipe.set_length(length)
    return datapipe, label_to_speaker_id


def voxceleb1_test_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    features1 = [data[1][0].T for data in batch]
    features1 = [(feature - feature.mean()) / (feature.std() + 1.e-9)
                for feature in features1]  # Instance norm
    features2 = [data[2][0].T for data in batch]
    features2 = [(feature - feature.mean()) / (feature.std() + 1.e-9)
                for feature in features2]  # Instance norm
    labels = torch.tensor([data[0] for data in batch], dtype=torch.int64)
    lengths1 = torch.tensor([feature.size(0) for feature in features1])
    lengths2 = torch.tensor([feature.size(0) for feature in features2])

    features1_padded = pad_sequence(features1, batch_first=True, padding_value=0)
    features2_padded = pad_sequence(features2, batch_first=True, padding_value=0)

    return features1_padded, lengths1, features2_padded, lengths2, labels


def build_voxceleb1_test_datapipe(voxceleb1_root, feature_extractor,
                                  augmentor: Callable = None,
                                  test_pair_file_path: str = "Vox1-O_pairs.txt",
                                  batch_size=1):
    def load_voxceleb1_test_pair(item, voxceleb1_root):
        label, audio_path1, audio_path2 = item
        # Load audio
        if Path(voxceleb1_root + "test/wav/" + audio_path1).exists():
            waveform1, _ = torchaudio.load(voxceleb1_root + "test/wav/" + audio_path1)
        else:
            waveform1, _ = torchaudio.load(voxceleb1_root + "dev/wav/" + audio_path1)

        if Path(voxceleb1_root + "test/wav/" + audio_path2).exists():
            waveform2, _ = torchaudio.load(voxceleb1_root + "test/wav/" + audio_path2)
        else:
            waveform2, _ = torchaudio.load(voxceleb1_root + "dev/wav/" + audio_path2)

        waveform1 = (waveform1 - waveform1.mean()) / (waveform1.std() + 1.e-9)
        waveform2 = (waveform2 - waveform2.mean()) / (waveform2.std() + 1.e-9)
        return int(label), waveform1, waveform2

    def get_paired_flist(veri_test_path: str):
        file_list = []
        with open(veri_test_path, "r", encoding="utf8") as file:
            for line in file:
                label, path1, path2 = line.split()
                file_list.append((label, path1, path2))
        return file_list

    def waveform_augment(item, augmentor):
        waveform = item
        waveform_augmented = augmentor(waveform.unsqueeze(0))[0]
        return waveform_augmented

    def extract_features(item, feature_extractor):
        waveform = item
        features = feature_extractor(waveform)
        return features

    verification_pairs = get_paired_flist(veri_test_path=test_pair_file_path)
    datapipe = dp.iter.IterableWrapper(verification_pairs)
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.map(partial(load_voxceleb1_test_pair, voxceleb1_root=voxceleb1_root))
    datapipe_label, datapipe_waveform1, datapipe_waveform2 = datapipe.unzip(sequence_length=3)

    if augmentor:
        datapipe_waveform1_augmented = datapipe_waveform1.map(partial(waveform_augment, augmentor=augmentor))
        datapipe_features1 = datapipe_waveform1_augmented.map(partial(extract_features,
                                                                      feature_extractor=feature_extractor))
        datapipe_waveform2_augmented = datapipe_waveform2.map(partial(waveform_augment, augmentor=augmentor))
        datapipe_features2 = datapipe_waveform2_augmented.map(partial(extract_features,
                                                                      feature_extractor=feature_extractor))
    else:
        datapipe_features1 = datapipe_waveform1.map(
            partial(extract_features, feature_extractor=feature_extractor))
        datapipe_features2 = datapipe_waveform2.map(
            partial(extract_features, feature_extractor=feature_extractor))

    datapipe = datapipe_label.zip(datapipe_features1, datapipe_features2)
    datapipe = datapipe.batch(batch_size=batch_size, drop_last=False)
    datapipe = datapipe.set_length(len(verification_pairs))
    return datapipe


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=pad_collate,
                    pin_memory: bool = False, batch_sampler=None):
    if batch_sampler:
        data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler,
                                 num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    else:
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

    librispeech_root = "/Users/JG96XG/Desktop/data_sets/LibriSpeech/sharded/raw/"  # "/Users/holge/Downloads/LibriSpeech/"
    librispeech_splits = ["train-clean-100"]
    # voxceleb1_root = "/Users/JG96XG/Desktop/data_sets/VoxCeleb/voxceleb_1/"
    # voxceleb1_splits = ["test"]
    # voxceleb2_root = "/Users/JG96XG/Desktop/data_sets/VoxCeleb/voxceleb_2/"
    # voxceleb2_splits = ["test"]

    data_sets = {"librispeech": {"root": librispeech_root, "splits": librispeech_splits}}


    def augmentor(waveform):
        return waveform ** 2

    datapipe, label_to_speaker_id = build_xvector_datapipe(data_sets,
                                                           feature_extractor=feature_extractor,
                                                           buffer_size=1000,
                                                           load_from="raw",
                                                           load_from_tar=True,
                                                           augmentor=augmentor,
                                                           max_token_count=10000,
                                                           segment_max_size=500,
                                                           min_length=50)


    # voxceleb1_root = "/Users/JG96XG/Desktop/data_sets/VoxCeleb/voxceleb_1/"
    # voxceleb1_splits = ["test"]
    # test_data_sets = {"voxceleb": {"root": voxceleb1_root, "splits": voxceleb1_splits}}
    # test_pair_file_path = "/Users/JG96XG/Desktop/data_sets/VoxCeleb/test_lists/Vox1-O_pairs.txt"
    #
    # test_datapipe = build_voxceleb1_test_datapipe(voxceleb1_root=voxceleb1_root,
    #                                               feature_extractor=feature_extractor,
    #                                               augmentor=augmentor,
    #                                               test_pair_file_path=test_pair_file_path)
    from pathlib import Path
    it1 = iter(datapipe)
    dataloader = get_data_loader(datapipe, collate_fn=pad_collate, batch_size=1, num_workers=0)
    for data in dataloader:
        feats, lengths, labels = data

    print("done")
