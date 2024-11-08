from typing import Tuple, List
from math import prod

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram


class LogMel(nn.Module):
    def __init__(self, window_length=400, hop_length=160, n_fft=400, n_mels=40, sample_rate=16000):
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mel_spectrogram = MelSpectrogram(sample_rate=self.sample_rate,
                                              n_fft=self.n_fft,
                                              n_mels=self.n_mels,
                                              win_length=self.window_length,
                                              hop_length=self.hop_length)
        self.feature_rate = round(sample_rate / hop_length)

    def forward(self, x: torch.Tensor):
        x = self.mel_spectrogram(x)
        x = torch.log10(x + 1.e-6)
        return x
