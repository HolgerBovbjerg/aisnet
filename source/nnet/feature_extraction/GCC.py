from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .STFT import STFT


def add_noise_to_waveform(waveforms: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add noise to a batch of waveform tensors at a specific SNR (signal-to-noise ratio).

    Args:
        waveforms (torch.Tensor): Batch of waveform tensors of shape (batch_size, n_samples).
        snr_db (float): Desired SNR in decibels (dB).

    Returns:
        torch.Tensor: Batch of waveforms with added noise at the specified SNR.
    """
    # Compute signal power for each waveform (mean square value along the sample axis)
    signal_power = torch.mean(waveforms ** 2, dim=1, keepdim=True)

    # Convert SNR from decibels to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power for each waveform based on the desired SNR
    noise_power = signal_power / snr_linear

    # Generate white Gaussian noise for each waveform with the calculated noise power
    noise = torch.randn_like(waveforms) * torch.sqrt(noise_power)

    # Add noise to each waveform in the batch
    noisy_waveforms = waveforms + noise

    return noisy_waveforms


def apply_phase_transform(cps: torch.Tensor) -> torch.Tensor:
    # Compute the magnitude of the cross power spectrum
    magnitude = torch.abs(cps)

    # Phase transform
    return cps / (magnitude + 1.e-8)


def estimate_doa(tdoa: torch.Tensor, dist_mics: float, c_sound: float = 343.):
    dist_wave = c_sound * tdoa
    dist_wave[dist_wave >= dist_mics] = dist_mics - 1.e-7
    dist_wave[dist_wave <= -dist_mics] = -dist_mics + 1.e-7
    return torch.arcsin(dist_wave / dist_mics) * (180. / torch.pi)

@dataclass
class GCCConfig:
    n_fft: int = 512
    window_length: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    center: bool = False
    window_type: str = "hann"
    max_delay: Optional[torch.Tensor] = None
    n_mics: int = 2
    phase_transform: bool = True


class GCC(nn.Module):
    def __init__(self, n_fft: int, window_length: int, hop_length: int, sample_rate: int, center: bool = False,
                 window_type: str = "hann", max_delay: Optional[torch.Tensor] = None, n_mics: int = 2,
                 phase_transform: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_length = window_length
        self.n_fft = n_fft
        self.center = center
        self.window_type = window_type
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, window_length=window_length, sample_rate=sample_rate,
                         center=center, window_type=window_type)
        self.max_delay = max_delay if max_delay is not None else [n_fft // 2 + 1]
        self.delays = [torch.arange(-delay, delay + 1, dtype=torch.int) for delay in self.max_delay]
        self.n_mics = n_mics
        self.mic_indices_i, self.mic_indices_j = torch.triu_indices(self.n_mics, self.n_mics, offset=1)
        self.mic_pairs = tuple((int(i), int(j)) for i, j in zip(self.mic_indices_i, self.mic_indices_j))
        self.phase_transform = phase_transform
        self.apply_phase_transform = apply_phase_transform if phase_transform else lambda x: x

    def forward(self, input_tensor: torch.Tensor, output_delay: bool = False, output_tdoa: bool = False):
        # Expecting input_tensor shape: (batch, channels, time)
        stft = self.stft(input_tensor)

        # Gather the necessary STFT data for each microphone pair
        stft_i = stft[:, self.mic_indices_i]  # (batch, num_pairs, time, stft_bins)
        stft_j = stft[:, self.mic_indices_j]  # (batch, num_pairs, time, stft_bins)

        # Compute the cross-power spectrum(CPS) for each pair of microphones
        cps = stft_j * stft_i.conj()  # (batch, num_pairs, time, stft_bins)

        # Phase transform (no-op if self.phase_transform is False)
        cps = self.apply_phase_transform(cps)

        # Inverse fft of CPS to go to time domain
        r = torch.fft.irfft(cps, dim=-1, n=self.n_fft).transpose(-1, -2)

        # max delay for each microphone pair is provided
        output = []
        for i, delay_range in enumerate(self.delays):
            output.append(r[:, i, delay_range])
        if output_delay:
            if output_tdoa:
                output = [self.estimate_tdoa(x) for x in output]
            else:
                output = [self.estimate_delay(x) for x in output]
        if len(output) == 1:
            return output[0]
        return output

    def estimate_delay(self, r: List[torch.Tensor]):
        delays = []
        for i, gcc in enumerate(r):
            delays.append(self.delays[i][torch.argmax(gcc, dim=-1)])
        return delays

    def estimate_tdoa(self, r: torch.Tensor):
        delay_samples = self.estimate_delay(r)
        tdoa = [delay / self.sample_rate for delay in delay_samples]
        return tdoa


if __name__ == "__main__":
    from math import ceil, floor
    import copy

    import torchaudio

    torch.manual_seed(0)
    # Sample rate and maximum delay in seconds
    sample_rate = 16000

    speech_signal, fs = torchaudio.load('/Users/JG96XG/PycharmProjects/aisnet/data/OSR_us_000_0010_8k.wav')
    if fs != sample_rate:
        speech_signal = torchaudio.functional.resample(speech_signal, fs, sample_rate)

    T = 1
    length = T * sample_rate
    speech_signal = speech_signal[:, :length]
    delay_samples_true = 3
    delay_time_true = delay_samples_true / sample_rate
    input_tensor = torch.stack([speech_signal, speech_signal, speech_signal, speech_signal], dim=0)
    if delay_samples_true > 0:
        left = input_tensor
        right = copy.deepcopy(input_tensor).roll(abs(delay_samples_true))
    else:
        right = input_tensor
        left = copy.deepcopy(input_tensor).roll(abs(delay_samples_true))

    input_tensor = torch.concat([left, right], dim=1)
    #input_tensor = add_noise_to_waveform(input_tensor, snr_db=20)

    microphone_positions = torch.tensor([[0, -0.09, 0], [0, 0.09, 0]])
    # Compute pairwise distances
    diff = microphone_positions.unsqueeze(1) - microphone_positions.unsqueeze(0)
    dist_mics = torch.sqrt(torch.sum(diff ** 2, dim=-1))
    indices = torch.triu_indices(*dist_mics.size(), offset=1)
    dist_mics = dist_mics[indices[0], indices[1]]
    c_sound = 343
    t_max = dist_mics / c_sound
    t_max_samples = torch.floor(t_max * sample_rate)
    resolution = torch.arcsin(torch.tensor(c_sound * (1 / sample_rate) / dist_mics)) * (180. / torch.pi)

    # Instantiate the GCC-PHAT module
    gcc_phat = GCCPHAT(n_fft=512, window_length=401, hop_length=160, sample_rate=sample_rate, center=True,
                       max_delay=t_max_samples, n_mics=microphone_positions.shape[0])

    # Compute the time delay using GCC-PHAT
    r = gcc_phat(input_tensor)
    delay = gcc_phat.estimate_delay(r)
    tdoa = gcc_phat.estimate_tdoa(r)
    doa = estimate_doa(tdoa=tdoa[0], dist_mics=dist_mics[0], c_sound=c_sound)
    lowpass_kernel_size = 11
    doa_lowpass = \
        F.avg_pool1d(doa, kernel_size=lowpass_kernel_size, stride=1, padding=lowpass_kernel_size // 2)[0]
    doa_real = estimate_doa(tdoa=torch.tensor(delay_time_true), dist_mics=dist_mics[0], c_sound=c_sound)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(2, 1)
    delay_range = torch.tensor([-t_max_samples[0], t_max_samples[0]]) * resolution[0]
    ax[0].imshow(r[0][0].T, extent=(0, doa.size(-1), delay_range[0], delay_range[1]), aspect='auto')
    ax[1].plot(doa[0], label='raw estimate')
    ax[1].plot(doa_lowpass[0], label='lowpass estimate')
    ax[1].plot(doa_real * torch.ones(doa.size(-1)), label='real', linewidth=2)
    ax[1].set_ylim([delay_range[0], delay_range[1]])
    ax[1].legend()

    ax[1].set_xlabel(f"Time [degree]")
    ax[0].set_ylabel(f"Azimuth angle [degree]")
    ax[1].set_ylabel(f"Azimuth angle [degree]")

    fig.suptitle("GCC-PHAT estimation")
    fig.show()
    print("Done")
