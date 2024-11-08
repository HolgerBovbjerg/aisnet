from pathlib import Path
from functools import partial
import random
from io import BytesIO
from typing import Callable, Iterator, Optional, Sized, TypeVar

import torch
from torch.nn.utils.rnn import pad_sequence
from torchdata import datapipes as dp
import torchaudio
import numpy as np
from rVADfast import rVADfast
import torch.utils.data.datapipes.iter.sharding
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

from common.misc import nearest_interp

T_co = TypeVar("T_co", covariant=True)


@functional_datapipe('unique_id_batch')
class UniqueIdBatchIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe[T_co]
    max_buffer_size: int
    id_fn: Callable
    max_buffer_size: int
    current_buffer_size: int
    batch_size: int
    id_fn: Callable
    _buffer: dict[T_co]

    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 id_fn: Callable,
                 *,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 unbatch_level: int = 0,
                 ) -> None:
        super().__init__()
        _check_unpickable_fn(id_fn)
        self._buffer: dict = {}
        assert buffer_size > 0, "buffer_size should be larger than 0"
        if unbatch_level == 0:
            self.datapipe = datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)
        self.max_buffer_size = buffer_size
        self.current_buffer_size = 0
        self.batch_size = batch_size
        self.id_fn = id_fn

    def add_element(self, element_id, data):
        if element_id in self._buffer:
            self._buffer[element_id].append(data)
        else:
            self._buffer[element_id] = [data]
        self.current_buffer_size += 1

    def sample(self, n):
        sampled_elements = []
        sampled_ids = set()
        while len(sampled_elements) < n:
            element_id = random.choice(list(self._buffer.keys()))
            if element_id not in sampled_ids:
                sampled_ids.add(element_id)
                sampled_data = self._buffer[element_id].pop()
                sampled_elements.append(sampled_data)
                if not self._buffer[element_id]:
                    del self._buffer[element_id]
                self.current_buffer_size -= 1
        return sampled_elements

    def pop_element(self):
        # Find the ID with the most samples
        id_with_most_samples = max(self._buffer, key=lambda x: len(self._buffer[x]))
        # Pop an element from the ID with the most samples
        self._buffer[id_with_most_samples].pop()
        # If the ID is empty after popping, remove it from the buffer
        if not self._buffer[id_with_most_samples]:
            del self._buffer[id_with_most_samples]

    def __iter__(self) -> Iterator[T_co]:
        for x in self.datapipe:
            id = self.id_fn(x)
            if self.current_buffer_size == self.max_buffer_size:
                self.pop_element()
            self.add_element(id, x)
            if len(self._buffer) >= self.batch_size:
                yield self.sample(self.batch_size)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

    def __getstate__(self):
        state = (
            self.datapipe,
            self.max_buffer_size,
            self.current_buffer_size,
            self.batch_size,
            self.id_fn,
            self._buffer,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.max_buffer_size,
            self.current_buffer_size,
            self.batch_size,
            self.id_fn,
            self._buffer,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state

    def __del__(self):
        self._buffer.clear()


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
    if ".flac" in wds_item.keys():
        audio_ext = ".flac"
    elif ".wav" in wds_item.keys():
        audio_ext = ".wav"
    elif ".wav.wav" in wds_item.keys():
        audio_ext = ".wav.wav"
    else:
        raise ValueError("no flac or wav file in wds_keys")

    if load_from == "raw":
        return wds_item["__key__"], wds_item[audio_ext]
    elif load_from == "decoded":
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


def segment_features(item, segment_size, randomize_start: bool = False, drop_last: bool = False):
    features, speaker_id, audio_path = item
    length = features.size(-1)
    n_segments = length // segment_size
    if randomize_start:
        remainder = length % segment_size
        start = random.randint(0, remainder)
    else:
        start = 0
    features = features[:, :, start:].split(segment_size, dim=-1)
    if drop_last and (len(features) > n_segments):
        features = features[:-1]

    output = [(feature, speaker_id, audio_path, segment_id) for segment_id, feature in enumerate(features)]
    return output


def group_fn(item):
    speaker_id = item[1]
    return speaker_id


def id_fn(item):
    speaker_id = item[0][1]
    return speaker_id


def build_ge2e_datapipe(data_sets, batch_size, n_utterances, feature_extractor=None, augmentor=None,
                        chunk_size=None, buffer_size: int = 10000, load_from: str = "raw",
                        shuffle=True, load_from_tar: bool = False, remove_non_speech: bool = False):
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
        if remove_non_speech:
            vad = rVADfast()
            datapipe = datapipe.map(partial(remove_non_speech, vad=vad))
        if augmentor:
            datapipe = datapipe.map(partial(augment, augmentor=augmentor))
        datapipe = datapipe.map(partial(extract_features, feature_extractor=feature_extractor))

    # Segment into max(chunk_size) slices
    datapipe = datapipe.flatmap(partial(segment_features, segment_size=max(chunk_size),
                                        randomize_start=True, drop_last=True))

    # Shuffle after segmentation
    # datapipe = datapipe.shuffle(buffer_size=buffer_size)

    # Group features by speaker id, batch and collate
    datapipe = datapipe.groupby(group_key_fn=group_fn, group_size=n_utterances, guaranteed_group_size=n_utterances,
                                buffer_size=buffer_size, drop_remaining=True)
    datapipe = datapipe.unique_id_batch(buffer_size=buffer_size, batch_size=batch_size, id_fn=id_fn)
    #       .collate(partial(collate_fn, chunk_size=chunk_size))

    return datapipe


def chunk_features(features, chunk_size=(160, 180)):
    low, high = chunk_size
    chunk_length = random.randint(low, high)
    start_points = np.array([random.randint(0, feature.size(0) - chunk_length)
                             for feature in features])
    end_points = start_points + chunk_length
    features_chunked = [feature[start: end] for feature, start, end in zip(features, start_points, end_points)]
    lengths = torch.ones(len(features_chunked)) * chunk_length
    return features_chunked, lengths


def collate_fn(batch, chunk_size=None):
    features = [item[0][0].T for speaker_batch in batch for item in speaker_batch]
    features = [(feature - feature.mean()) / (feature.std() + 1.e-9)
                for feature in features]  # Instance norm
    if chunk_size:
        features, lengths = chunk_features(features, chunk_size)
        features = pad_sequence(features, batch_first=True)
    else:
        lengths = torch.tensor([feature.size(0) for feature in features])
        features = pad_sequence(features, batch_first=True)
    speaker_ids = [item[1] for speaker_batch in batch for item in speaker_batch]
    audio_paths = [item[2] for speaker_batch in batch for item in speaker_batch]
    segment_ids = [item[3] for speaker_batch in batch for item in speaker_batch]
    return features, lengths, speaker_ids, audio_paths, segment_ids


def get_ge2e_loader(datapipe, reading_service: str = "multiprocessing", batch_size: int = 1, num_workers: int = 0,
                    pin_memory=False, collate_fn=collate_fn, shuffle=True, chunk_size=None):
    return torch.utils.data.DataLoader(dataset=datapipe, num_workers=num_workers, pin_memory=pin_memory,
                                       collate_fn=partial(collate_fn, chunk_size=chunk_size), batch_size=batch_size,
                                       shuffle=shuffle)
    # rs = None
    # if reading_service == "multiprocessing":
    #     rs = MultiProcessingReadingService(num_workers=num_workers)
    # elif reading_service == "distributed":
    #     rs = DistributedReadingService()
    # elif reading_service == "distributed_multiprocessing":
    #     mp_rs = MultiProcessingReadingService(num_workers=num_workers)
    #     dist_rs = DistributedReadingService()
    #     rs = SequentialReadingService(dist_rs, mp_rs)

    # return DataLoader2(datapipe, reading_service=rs)


if __name__ == "__main__":
    # from src.feature_extraction import FeatureExtractor

    # librispeech_root = "/scratch/project_465000676/Data/LibriSpeech/"
    librispeech_root = "/Users/JG96XG/Desktop/data_sets/LibriSpeech/"
    librispeech_splits = ["dev-clean", "train-clean-100"]
    voxceleb1_root = "/scratch/project_465000676/VoxCeleb/voxceleb_1/"
    voxceleb1_splits = ["test"]
    voxceleb2_root = "/scratch/project_465000676/VoxCeleb/voxceleb_2/"
    voxceleb2_splits = ["test"]

    data_sets = {"librispeech": {"root": librispeech_root, "splits": librispeech_splits}}  # ,
    # "voxceleb1": {"root": voxceleb1_root, "splits": voxceleb1_splits}}#,
    # "voxceleb2": {"root": voxceleb2_root, "splits": voxceleb2_splits}}
    # data_sets = {"voxceleb1": {"root": voxceleb1_root, "splits": voxceleb1_splits}}

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

    chunk_size = (140, 180)
    datapipe = build_ge2e_datapipe(data_sets, n_utterances=10, batch_size=64, feature_extractor=feature_extractor,
                                   buffer_size=1000, load_from="raw", chunk_size=chunk_size,
                                   load_from_tar=False)

    dataloader = get_ge2e_loader(datapipe, collate_fn=collate_fn, chunk_size=chunk_size, batch_size=1)
    for data in dataloader:
        features, lengths, speaker_ids, audio_paths, segment_ids = data
        print(data)
    print("done")
