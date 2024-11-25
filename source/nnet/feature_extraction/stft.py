from dataclasses import dataclass

import torch
from PIL.ImageOps import posterize
from torch import nn
from torch.nn import functional as F


@dataclass
class STFTConfig:
    n_fft: int = 512
    window_length: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    center: bool = True
    window_type: str = "hann"
    pad_mode: str = "reflect"
    output_type: str = "raw"


class STFT(nn.Module):
    def __init__(self, n_fft: int, window_length: int, hop_length: int, sample_rate: int, center: bool = True,
                 window_type: str = "hann", pad_mode: str = "reflect", output_type: str = "raw"):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_length = window_length
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(window_length))
        self.center = center
        self.pad_mode = pad_mode
        # Dictionary of supported window functions
        window_functions = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window
        }

        # Generate the chosen window
        if window_type not in window_functions:
            raise ValueError(
                f"Unsupported window type: {window_type}. Supported types are: {list(window_functions.keys())}")
        self.register_buffer("window", window_functions[window_type](window_length))
        self.postprocessing_function = self._select_postprocessing_function(output_type)

    def _select_postprocessing_function(self, output_type: str):
        """
        Returns the processing function based on the output_type.
        """
        postprocessing_func_map = {
            "raw": self._raw,
            "power_phase": self._power_phase,
            "power": self._power,
            "log_power_phase": self._log_power_phase,
            "log_power": self._log_power,
        }

        if output_type not in postprocessing_func_map:
            raise ValueError(
                f"Unsupported output_type: {output_type}. Supported types are: {list(postprocessing_func_map.keys())}")

        return postprocessing_func_map[output_type]

    @staticmethod
    def _raw(stft_output):
        return stft_output

    @staticmethod
    def _power(stft_output):
        return torch.abs(stft_output) ** 2

    def _log_power(self, stft_output):
        eps = 1e-10
        power = self._power(stft_output)
        log_power = torch.log10(power + eps)
        return log_power

    @staticmethod
    def _phase(stft_output):
        return torch.angle(stft_output)

    def _power_phase(self, stft_output):
        power = self._power(stft_output)
        phase = self._phase(stft_output)
        return torch.cat((power, phase), dim=-1)

    def _log_power_phase(self, stft_output):
        log_power = self._log_power(stft_output)
        phase = self._phase(stft_output)
        return torch.cat((log_power, phase), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expecting input_tensor shape: (batch, channels, time)
        if self.center:
            x = F.pad(x, (self.window_length // 2, self.window_length // 2),
                      mode=self.pad_mode)

        # Window input
        x = x.unfold(dimension=-1, size=self.window_length,
                     step=self.hop_length) * self.window

        # Compute FFT along the window size dimension (last dimension)
        x = torch.fft.rfft(x, dim=-1, n=self.n_fft)

        return self.postprocessing_function(x)
