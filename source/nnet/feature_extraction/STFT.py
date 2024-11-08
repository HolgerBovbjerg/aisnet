import torch
from torch import nn
from torch.nn import functional as F


class STFT(nn.Module):
    def __init__(self, n_fft: int, window_length: int, hop_length: int, sample_rate: int, center: bool = False,
                 window_type: str = "hann", pad_mode: str = "constant"):
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
        self.window = window_functions[window_type](window_length)

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

        return x
