from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram


@dataclass
class LogMelConfig:
    n_mel: int = 40
    window_length: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    center: bool = True
    window_type: str = "hann"
    pad_mode: str = "reflect"
    normalized: bool = False
    norm: Optional[str] = None
    mel_scale: str = "htk"


class LogMel(nn.Module):
    def __init__(self, window_length=400, hop_length=160, n_fft=400, n_mels=40, sample_rate=16000,
                 window_type: str = "hann", normalized: bool = False, center: bool = True, norm: Optional[str] = None,
                 mel_scale: str ="htk", pad_mode: str = "reflect",):
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        # Dictionary of supported window functions
        window_functions = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window
        }
        self.mel_spectrogram = MelSpectrogram(sample_rate=self.sample_rate,
                                              n_fft=self.n_fft,
                                              n_mels=self.n_mels,
                                              win_length=self.window_length,
                                              hop_length=self.hop_length,
                                              window_fn=window_functions[window_type],
                                              normalized=normalized,
                                              center=center,
                                              norm=norm,
                                              pad_mode=pad_mode,
                                              mel_scale=mel_scale)
        self.feature_rate = round(sample_rate / hop_length)

    def forward(self, x: torch.Tensor):
        x = self.mel_spectrogram(x)
        x = torch.log10(x + 1.e-6)
        return x
