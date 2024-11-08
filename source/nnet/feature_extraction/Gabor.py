from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import melscale_fbanks


class RealImagGaborFilter(nn.Module):
    def __init__(self, num_filters: int, n_coefficients: int, initial_center_frequencies: torch.Tensor,
                 initial_inverse_bandwidths: torch.Tensor, causal: bool = False, sample_rate: float = 16000.,
                 stride: int = 1):
        super().__init__()
        # Create timescale for filters
        self.filter_length = n_coefficients
        self.num_filters = num_filters
        self.n_coefficients = n_coefficients
        self.causal = causal
        self.sample_rate = sample_rate
        self.stride = stride

        t = torch.arange(-self.n_coefficients // 2, (self.n_coefficients + 1) // 2).float()
        self.register_buffer('t', t.unsqueeze(0))

        if self.causal:
            self.padding = self.filter_length - 1
        else:
            self.padding = self.filter_length // 2

        # Learnable parameters for center frequencies and bandwidths
        self.center_frequencies = nn.Parameter(initial_center_frequencies)
        self.bandwidths = nn.Parameter(initial_inverse_bandwidths)

    def gabor_impulse_response(self):
        # Compute Gaussian normalization factor
        normalization = 1.0 / (torch.sqrt(torch.tensor(2.0) * torch.pi) * self.bandwidths).unsqueeze(-1)

        # Compute the Gaussian envelope
        gaussian = torch.exp((-self.t ** 2) / (2.0 * (self.bandwidths.unsqueeze(-1) ** 2)))

        # Compute the complex sinusoid (carrier wave)
        sinusoid_real = torch.cos(self.center_frequencies.unsqueeze(-1) * self.t)
        sinusoid_imag = torch.sin(self.center_frequencies.unsqueeze(-1) * self.t)

        # Compute filters
        real_filter = normalization * gaussian * sinusoid_real
        imag_filter = normalization * gaussian * sinusoid_imag

        # Return the Gabor impulse response (Gaussian * Sinusoid * Normalization)
        return real_filter, imag_filter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get number of channels
        batch_size = x.size(0)
        n_channels = x.size(1)

        # If multiple channels are present, the channels are stacked in the batch dimension
        if n_channels > 1:
            x = x.reshape(batch_size * n_channels, 1, x.size(2))

        # Get gabor filter
        real_filter, imag_filter = self.gabor_impulse_response()
        real_filter = real_filter.unsqueeze(1)
        imag_filter = imag_filter.unsqueeze(1)

        # Combine real and imaginary filters into 2N filters
        real_imag_filter = torch.cat((real_filter, imag_filter), dim=0)
        if self.causal:
            # Pad the input on the left (start) with (kernel_size - 1) zeros
            x = F.pad(x, (self.padding, 0))

        # Convolve signal and gabor filters
        x = F.conv1d(x, real_imag_filter, stride=self.stride)

        # Combine real and imaginary part into complex tensor
        x = x[:, :self.num_filters] + 1.j * x[:, self.num_filters:]

        # Split channels if necessary
        if n_channels > 1:
            x = x.reshape(batch_size, n_channels, x.size(1), x.size(2))

        return x


class ComplexGaborFilter(nn.Module):
    def __init__(self, num_filters: int, n_coefficients: int, initial_center_frequencies: torch.Tensor,
                 initial_inverse_bandwidths: torch.Tensor, causal: bool = False, sample_rate: float = 16000.,
                 stride: int = 1):
        super().__init__()
        self.filter_length = n_coefficients
        self.num_filters = num_filters
        self.n_coefficients = n_coefficients
        self.causal = causal
        self.sample_rate = sample_rate
        self.stride = stride

        # Create timescale for filters
        t = torch.arange(-self.n_coefficients // 2, self.n_coefficients // 2 + 1).float()
        self.register_buffer('t', t.unsqueeze(0))

        if self.causal:
            self.padding = self.filter_length - 1
        else:
            self.padding = self.filter_length // 2

        # Learnable parameters for center frequencies and bandwidths
        self.center_frequencies = nn.Parameter(initial_center_frequencies)
        self.bandwidths = nn.Parameter(initial_inverse_bandwidths)

    def gabor_impulse_response(self):
        # Compute Gaussian normalization factor
        normalization = 1.0 / (torch.sqrt(torch.tensor(2.0) * torch.pi) * self.bandwidths).unsqueeze(-1)

        # Compute the Gaussian envelope
        gaussian = torch.exp((-self.t ** 2) / (2.0 * (self.bandwidths.unsqueeze(-1) ** 2)))

        # Compute the complex sinusoid (carrier wave)
        sinusoid = torch.exp(1j * self.center_frequencies.unsqueeze(-1) * self.t)

        # Return the Gabor impulse response (Gaussian * Sinusoid * Normalization)
        return normalization * sinusoid * gaussian

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get number of channels
        batch_size = x.size(0)
        n_channels = x.size(1)

        # If multiple channels are present, the channels are stacked in the batch dimension
        if n_channels > 1:
            x = x.reshape(batch_size * n_channels, 1, x.size(2))

        # Apply complex Gabor filters to the input signal
        x = x.to(torch.complex64)  # Ensure input is complex
        if self.causal:
            # Pad the input on the left (start) with (kernel_size - 1) zeros
            x = F.pad(x, (self.padding, 0))

        filters = self.gabor_impulse_response().unsqueeze(1)

        # Complex valued convolution
        x = F.conv1d(x, filters, stride=self.stride, padding=0)

        # Split channels if necessary
        if n_channels > 1:
            x = x.reshape(batch_size, n_channels, x.size(1), x.size(2))
        return x


class GaborLayer(nn.Module):
    """
    GaborLayer applies a bank of Gabor filters to the input waveform. These filters
    are initialized with frequencies spanning a range of interest (e.g., from 60 Hz to 7800 Hz)
    and have learnable center frequencies and bandwidths.

    Args:
        num_filters (int): Number of Gabor filters to apply.
        sample_rate (int): The sampling rate of the input waveform.
        min_frequency (float): Minimum frequency for the Gabor filters.
        max_frequency (float): Maximum frequency for the Gabor filters.
    """

    def __init__(self, num_filters: int = 40, n_coefficients: int = 401, sample_rate: int = 16000,
                 min_frequency: float = 60.0, max_frequency: float = 7800.0, filter_init_method: str = "mel",
                 use_complex_convolution: bool = False, causal: bool = False, stride: int = 1):
        super().__init__()
        self.num_filters = num_filters
        self.n_coefficients = n_coefficients
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.filter_init_method = filter_init_method
        self.use_complex_convolution = use_complex_convolution
        self.causal = causal

        # Initialize center frequencies (either linearly spaced or Mel scale)
        if filter_init_method == "mel":
            initial_center_frequencies, initial_inverse_bandwidths = self.get_mel_init_parameters()
        elif filter_init_method == "linear":
            n = np.linspace(0, 1, num_filters)
            initial_center_frequencies = min_frequency + (max_frequency - min_frequency) * n
            bandwidth = n_coefficients * torch.sqrt(2 * torch.log(torch.tensor(2.)))
            initial_inverse_bandwidths = torch.ones(num_filters) * bandwidth
        else:
            raise ValueError(f"Invalid filter initialization method {filter_init_method}")

        self.register_buffer("min_bandwidth",
                             4 * torch.sqrt(2 * torch.log(torch.tensor(2.))) / torch.pi)
        self.register_buffer("max_bandwidth",
                             2 * n_coefficients * torch.sqrt(2 * torch.log(torch.tensor(2.))) / torch.pi)
        self.register_buffer('min_center_frequency', torch.tensor(0.))
        self.register_buffer('max_center_frequency', torch.tensor(torch.pi))

        if self.use_complex_convolution:
            # Use module with complex numbered convolution
            self.gabor_filter = ComplexGaborFilter(num_filters=num_filters, n_coefficients=n_coefficients,
                                                   initial_center_frequencies=initial_center_frequencies,
                                                   causal=causal, sample_rate=sample_rate,
                                                   initial_inverse_bandwidths=initial_inverse_bandwidths,
                                                   stride=stride)
        else:
            # Filter real and imaginary part individually
            self.gabor_filter = RealImagGaborFilter(num_filters=num_filters, n_coefficients=n_coefficients,
                                                    initial_center_frequencies=initial_center_frequencies,
                                                    causal=causal, sample_rate=sample_rate,
                                                    initial_inverse_bandwidths=initial_inverse_bandwidths,
                                                    stride=stride)

    def get_mel_init_parameters(self):
        # Get Mel-scale triangular filters
        mel_scale_filters = melscale_fbanks(n_freqs=self.n_coefficients // 2 + 1, f_min=self.min_frequency,
                                            f_max=self.max_frequency,
                                            n_mels=self.num_filters, sample_rate=self.sample_rate)

        # Get peak value and frequency index of each filter
        sqrt_filters = torch.sqrt(mel_scale_filters)
        peaks, center_frequencies_index = torch.max(sqrt_filters, dim=0)

        # Find center frequency for each filter (normalized angular frequency)
        initial_center_frequencies = 2 * torch.pi * (center_frequencies_index / self.n_coefficients)

        # Find half-magnitude width for each filter and width
        half_magnitudes = peaks / 2.
        filter_half_magnitude_width = torch.sum((sqrt_filters >= half_magnitudes).float(), dim=0)

        # Compute inverse bandwidths
        max_bandwidth = torch.sqrt(2. * torch.log(torch.tensor(2.)))
        initial_inverse_bandwidths = max_bandwidth * self.n_coefficients / (filter_half_magnitude_width * torch.pi)
        return initial_center_frequencies, initial_inverse_bandwidths

    @torch.no_grad()
    def _clamp_bandwidth_and_center_frequencies(self):
        self.gabor_filter.bandwidths.data = torch.clamp(self.gabor_filter.bandwidths,
                                                        min=self.min_bandwidth,
                                                        max=self.max_bandwidth)
        self.gabor_filter.center_frequencies.data = torch.clamp(self.gabor_filter.center_frequencies,
                                                                min=0.,
                                                                max=self.max_center_frequency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GaborLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, num_samples).

        Returns:
            torch.Tensor: Filtered output of shape (batch_size, num_filters, num_samples).
        """

        # Complex convolution between filters and input
        self._clamp_bandwidth_and_center_frequencies()
        return self.gabor_filter(x)

    def visualize_filters(self, save_path: str = None, db_scale: bool = True, normalize: bool = True):
        """
        Visualizes the Gabor filters from a GaborLayer instance.

        Args:
            save_path (str, optional): If provided, saves the plot to this path. Otherwise, displays the plot.
            db_scale (bool, optional): If true, plots gabor_filter magnitude in db scale.
            normalize (bool, optional): If true, scales the filters between 0 and 1.
        """
        # Extract the Gabor filters
        num_filters = self.gabor_filter.num_filters

        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        fig.suptitle(f'Gabor Filters Visualization {"(Complex)" if self.use_complex_convolution else "(Real/Imag)"}', fontsize=16)

        t = self.gabor_filter.t.detach().numpy()[0, :] / self.sample_rate
        num_points = len(t)
        frequencies = np.fft.fftfreq(num_points, t[1] - t[0])
        plot_frequencies = frequencies[:num_points // 2]  # Keep only the positive frequencies
        colors = plt.cm.inferno(np.linspace(0, 1, num_filters))  # Use a colormap to differentiate filters

        if self.use_complex_convolution:
            gabor_filter = self.gabor_filter.gabor_impulse_response()
        else:
            filter_real, filter_imag = self.gabor_filter.gabor_impulse_response()
            gabor_filter = filter_real + 1.j * filter_imag

        gabor_filter = gabor_filter.detach().numpy()
        filter_fft = np.fft.fft(gabor_filter)

        for i in range(num_filters):
            # Compute the Fourier Transform
            filter_i_fft = 2 * filter_fft[i, :num_points // 2]  # Keep only the positive frequencies

            # Get amplitude and phase
            filter_i_mag = np.abs(filter_i_fft)
            if normalize:
                filter_i_mag = filter_i_mag / filter_i_mag.max()
            if db_scale:
                filter_i_mag = 20 * np.log10(filter_i_mag + 1e-10)  # Add a small value to avoid log(0)
            filter_i_phase = np.angle(filter_i_fft)
            color = colors[i]

            # Plot magnitude
            axes[0].plot(plot_frequencies, filter_i_mag, color=color, label=f'Filter {i + 1}')
            axes[0].set_title('Magnitude Spectrum')
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Magnitude [dB]')

            # Plot phase
            axes[1].plot(plot_frequencies, filter_i_phase, color=color, label=f'Filter {i + 1}')
            axes[1].set_title('Phase Spectrum')
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Phase')

        plt.tight_layout()  # Adjust layout to make room for suptitle
        if save_path:
            plt.savefig(save_path)
            print(f'Plot saved to {save_path}')
        else:
            plt.show()
