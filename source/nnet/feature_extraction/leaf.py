from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gabor import Gabor


class GaussianLowPass(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1,
                 causal=True, use_bias=True):
        super(GaussianLowPass, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        if self.causal:
            self.padding = self.kernel_size - 1
        else:
            self.padding = self.kernel_size // 2
        self.use_bias = use_bias
        self.in_channels = in_channels

        bandwidths = torch.ones((1, 1, in_channels, 1)) * 0.4
        # const init of 0.4 makes it approximate a Hann window
        self.bandwidths = nn.Parameter(bandwidths)
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.ones(in_channels))
        else:
            self.bias = None

        t = torch.arange(-self.kernel_size // 2, self.kernel_size // 2)
        t = torch.reshape(t, (1, self.kernel_size, 1, 1))
        self.register_buffer('t', t)

    @torch.no_grad()
    def _clamp_bandwidth(self):
        self.bandwidths.data = torch.clamp(self.bandwidths, min=(2. / self.kernel_size), max=0.5)

    def forward(self, x):
        # Limit bandwidth
        self._clamp_bandwidth()
        bandwidths = self.bandwidths

        # Compute gaussian kernel
        denominator = bandwidths * 0.5 * (self.kernel_size - 1)
        kernel = torch.exp(-0.5 * (self.t / denominator) ** 2)
        kernel = kernel.reshape(-1, self.kernel_size, self.in_channels)
        kernel = kernel.permute(2, 0, 1)

        if self.causal:
            # Pad the input on the left (start) with (kernel_size - 1) zeros
            x = F.pad(x, (self.padding, 0))

        # Convolution between input and Gaussian kernel
        outputs = F.conv1d(x, kernel, bias=self.bias, stride=self.stride, padding=0, groups=self.in_channels)
        return outputs


class ExponentialMovingAverage(nn.Module):
    def __init__(self, in_channels, coeff_init, per_channel: bool = False):
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        weights = torch.ones(in_channels, ) if self._per_channel else torch.ones(1, )
        self._weights = nn.Parameter(weights * self._coeff_init)

    @staticmethod
    def _scan(init_state, x, s):
        x = x.permute(2, 0, 1)
        accumulated = init_state
        results = [init_state.unsqueeze(0)]
        for sample in x[1:]:
            accumulated = (s * sample) + ((1.0 - s) * accumulated)
            results.append(accumulated.unsqueeze(0))
        results = torch.cat(results, dim=0)
        results = results.permute(1, 2, 0)
        return results

    @torch.no_grad()
    def _clamp_weights(self):
        self._weights.data = torch.clamp(self._weights, min=0., max=1.)

    def forward(self, x):
        self._clamp_weights()
        weights = self._weights
        initial_state = x[:, :, 0]
        ema = self._scan(initial_state, x, weights)
        return ema


class LogCompressionLayer(nn.Module):
    def __init__(self, initial_compression_factor: float = 2.0, eps: float = 1.e-6):
        super().__init__()
        self.compression_factor = nn.Parameter(torch.tensor(initial_compression_factor, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log10(x + self.eps)


class sPCENLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 alpha: float = 0.96,
                 smoothing_coefficient: float = 0.04,
                 delta: float = 2.0,
                 root: float = 2.0,
                 eps: float = 1e-6):
        super().__init__()
        self._alpha_init = alpha
        self._delta_init = delta
        self._root_init = root
        self.eps = eps
        self._smoothing_coefficient = smoothing_coefficient

        self.alpha = nn.Parameter(torch.ones(in_channels) * self._alpha_init)
        self.delta = nn.Parameter(torch.ones(in_channels) * self._delta_init)
        self.root = nn.Parameter(torch.ones(in_channels) * self._root_init)
        self.ema = ExponentialMovingAverage(in_channels, coeff_init=self._smoothing_coefficient, per_channel=True)

    def forward(self, x):
        alpha = torch.min(self.alpha, torch.tensor(1.0, dtype=x.dtype, device=x.device))
        root = torch.max(self.root, torch.tensor(1.0, dtype=x.dtype, device=x.device))
        ema_smoother = self.ema(x)
        exponent = 1. / root
        output = ((x / (self.eps + ema_smoother) ** alpha.view(1, -1, 1) + self.delta.view(1, -1, 1))
                  ** exponent.view(1, -1, 1) - self.delta.view(1, -1, 1) ** exponent.view(1, -1, 1))
        return output


@dataclass
class LEAFConfig:
    num_filters: int = 40
    n_coefficients: int = 401
    sample_rate: int = 16000,
    compression_factor: float = 0.96
    window_length: int = 401
    hop_length: int = 160,
    min_frequency: float = 60.0
    max_frequency: float = 7800.0
    filter_init_method: str = "mel"
    use_complex_convolution = True
    causal: bool = True
    compression: str = "sPCEN"


class LEAF(nn.Module):
    """
    LEAFFrontend implements the LEAF (Learnable Frontend for Audio) model,
    consisting of Gabor filters, Squeeze-and-Excitation layers,
    compression, temporal pooling, and normalization.

    Args:
        num_filters (int): Number of Gabor filters in the frontend.
        sample_rate (int): The sampling rate of the input waveform.
        compression_factor (float): Initial value for the compression exponent.
        pool_size (int): Size of the pooling window.
    """

    def __init__(self, num_filters: int = 40, n_coefficients: int = 401, sample_rate: int = 16000,
                 compression_factor: float = 0.96, window_length: int = 401, hop_length: int = 160,
                 min_frequency: float = 60.0, max_frequency: float = 7800.0, filter_init_method: str = "mel",
                 use_complex_convolution=True, causal: bool = True, compression: str = "sPCEN"):
        super().__init__()
        self.use_complex_convolution = use_complex_convolution
        self.gabor_layer = Gabor(num_filters=num_filters, n_coefficients=n_coefficients, sample_rate=sample_rate,
                                 min_frequency=min_frequency, max_frequency=max_frequency,
                                 filter_init_method=filter_init_method,
                                 use_complex_convolution=use_complex_convolution, causal=causal)
        self.pooling_layer = GaussianLowPass(in_channels=num_filters, kernel_size=window_length, stride=hop_length,
                                             causal=causal)
        if compression == "log":
            self.compression_layer = LogCompressionLayer(initial_compression_factor=compression_factor)
        elif compression == "sPCEN":
            self.compression_layer = sPCENLayer(in_channels=num_filters, root=compression_factor)
        self.normalization_layer = nn.InstanceNorm1d(num_features=num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LEAF frontend.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, num_samples).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_filters, time_steps).
        """
        x = self.gabor_layer(x)
        x = torch.abs(x) ** 2
        x = self.compression_layer(x)
        x = self.pooling_layer(x)
        x = self.normalization_layer(x)
        return x


if __name__ == '__main__':
    import torchaudio

    speech_signal, fs = torchaudio.load('/Users/JG96XG/PycharmProjects/BinauralSSL/data/OSR_us_000_0010_8k.wav')

    sample_rate = 16000
    if fs != sample_rate:
        speech_signal = torchaudio.functional.resample(speech_signal, fs, sample_rate)

    speech_signal = speech_signal[..., :5 * sample_rate]
    speech_signal = torch.stack([speech_signal, speech_signal], dim=0)

    # Assuming input is a batch of raw audio signals of shape [batch_size, 1, num_samples]
    num_filters = 40
    leaf = LEAF(num_filters=num_filters, sample_rate=sample_rate, filter_init_method="mel")
    leaf_complex = LEAF(num_filters=num_filters, sample_rate=sample_rate, use_complex_convolution=True,
                        filter_init_method="mel")

    features = leaf(speech_signal)
    features_2 = leaf_complex(speech_signal)

    leaf.gabor_layer.visualize_filters(db_scale=False, normalize=True)
    leaf_complex.gabor_layer.visualize_filters(db_scale=False, normalize=True)

    print(features.shape)  # Output shape will be [batch_size, num_filters, time_steps]
