from typing import Tuple, Optional
from math import ceil
from dataclasses import dataclass

import torch
import torch.nn as nn

from .STFT import STFT
from .GCC import apply_phase_transform


@dataclass
class SRPConfig:
    microphone_positions: torch.Tensor
    n_fft: int = 512
    window_length: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    center: bool = False
    window_type: str = 'hann'
    elevation_resolution: float = 5.
    azimuth_resolution: float = 5.
    elevation_range: Tuple[float, float] = (0., 180.)
    azimuth_range: Tuple[float, float] = (-180., 180.)
    c_sound: float = 343.
    frequency_range: Optional[Tuple[float, float]] = None

class SRP(nn.Module):
    def __init__(self, microphone_positions: torch.Tensor, n_fft: int = 512, window_length: int = 400,
                 hop_length: int = 160, sample_rate: int = 16000,
                 center: bool = False, window_type: str = 'hann',
                 elevation_resolution: float = 5., azimuth_resolution: float = 5.,
                 elevation_range: Tuple[float, float] = (0., 180.),
                 azimuth_range: Tuple[float, float] = (-180., 180.),
                 c_sound: float = 343.,
                 frequency_range: Optional[Tuple[float, float]] = None,
                 phase_transform: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_length = window_length
        self.n_fft = n_fft
        self.center = center
        self.stft = STFT(n_fft=n_fft, window_length=window_length, hop_length=hop_length, sample_rate=sample_rate,
                         center=center, window_type=window_type)
        self.n_samples = n_fft // 2 + 1
        self.microphone_positions = microphone_positions
        self.n_mics = self.microphone_positions.shape[0]
        # Create indices for the unique microphone pairs
        self.mic_indices_i, self.mic_indices_j = torch.triu_indices(self.n_mics, self.n_mics, offset=1)
        self.mic_pairs = tuple((int(i), int(j)) for i, j in zip(self.mic_indices_i,  self.mic_indices_j))
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.c_sound = c_sound
        self.frequency_range = frequency_range if frequency_range is not None else (0., sample_rate / 2)
        self.frequency_bin_range = [int(round(f / self.sample_rate * self.n_fft)) for f in self.frequency_range]
        self.frequency_bins = torch.arange(self.frequency_bin_range[0], self.frequency_bin_range[1], dtype=torch.int)
        self.elevation_resolution = elevation_resolution
        self.elevation_range = elevation_range
        self.azimuth_resolution = azimuth_resolution
        self.azimuth_range = azimuth_range
        self.azimuth_angles = None
        self.elevation_angles = None
        self.frequencies = sample_rate / n_fft * self.frequency_bins  # torch.arange(n_fft // 2 + 1)
        self.steering_vectors = self.compute_steering_vectors()
        self.steering_vector_to_angle_map = self.get_steering_vector_index_to_angle_mapping()
        self.time_delays = self.compute_inter_microphone_time_delays()
        self.phase_shifts = self.compute_phase_shifts()
        self.phase_transform = phase_transform
        self.apply_phase_transform = apply_phase_transform if phase_transform else lambda x: x

    def compute_steering_vectors(self):
        # Compute elevation and azimuth angles
        elevation_span = abs(self.elevation_range[1] - self.elevation_range[0])
        num_directions_elevation = ceil(elevation_span / self.elevation_resolution) if elevation_span else 1
        if elevation_span:
            elevation_angles = torch.arange(num_directions_elevation + 1) * self.elevation_resolution + \
                               self.elevation_range[0]
        else:
            elevation_angles = torch.tensor([self.elevation_range[0]])
        azimuth_span = abs(self.azimuth_range[1] - self.azimuth_range[0])
        num_directions_azimuth = ceil(azimuth_span / self.azimuth_resolution) if azimuth_span else 1
        if azimuth_span:
            azimuth_angles = torch.arange(num_directions_azimuth + 1) * self.azimuth_resolution + self.azimuth_range[0]
        else:
            azimuth_angles = torch.tensor([self.azimuth_range[0]])

        # Convert angles to radians
        self.elevation_angles = torch.deg2rad(elevation_angles)
        self.azimuth_angles = torch.deg2rad(azimuth_angles)

        # Create grid of steering vector angles elevation and azimuth
        theta, phi = torch.meshgrid(self.elevation_angles, self.azimuth_angles, indexing="xy")

        # Compute the cartesian steering vector coordinates (spherical coordinates to cartesian)
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        # Stack steering vectors
        steering_vectors = torch.stack((x, y, z), dim=-1)
        return steering_vectors.reshape(-1, 3)

    def get_steering_vector_index_to_angle_mapping(self):
        x, y, z = self.steering_vectors[..., 0], self.steering_vectors[..., 1], self.steering_vectors[..., 2]
        # Compute angles from the steering vectors
        computed_azimuth_angles = torch.rad2deg(torch.atan2(y, x))
        computed_elevation_angles = torch.rad2deg(torch.acos(z))
        angle_map = torch.stack((computed_elevation_angles, computed_azimuth_angles), dim=-1)
        return angle_map  # (last dimension of angle map is (azimuth, elevation)))

    def compute_inter_microphone_time_delays(self):
        # Loop over each microphone pair
        time_delays = torch.zeros(size=(self.n_mics, self.steering_vectors.size(0)), dtype=torch.float32)
        for i in range(self.n_mics):
            # Compute distance between selected microphone pair
            microphone_dist = (self.microphone_positions[i] * self.steering_vectors).sum(-1)
            # Divide by speed of sound to get difference in travel time
            time_delay = microphone_dist / self.c_sound
            time_delays[i] = time_delay
        return time_delays

    def compute_phase_shifts(self):
        # Convert frequencies to angular frequencies
        angular_frequencies = 2 * torch.pi * self.frequencies
        # For each microphone, compute phase shift (relative to origin) of each frequency for each direction
        relative_phase_shifts = torch.exp(1j * angular_frequencies.unsqueeze(0).unsqueeze(1)
                                          * self.time_delays.unsqueeze(-1))
        # Compute phase shifts for each microphone pair.
        phase_shifts = relative_phase_shifts[self.mic_indices_j].conj() * relative_phase_shifts[self.mic_indices_i]
        return phase_shifts

    def forward(self, input_tensor: torch.Tensor):
        # Expecting input_tensor shape: (batch, channels, time)
        stft = self.stft(input_tensor)[..., self.frequency_bins]

        # Gather the necessary STFT data for each microphone pair
        stft_i = stft[:, self.mic_indices_i]  # (batch, num_pairs, time, stft_bins)
        stft_j = stft[:, self.mic_indices_j]  # (batch, num_pairs, time, stft_bins)

        # Compute the cross-power spectra (CPS) for each pair of microphones
        cps = stft_j * stft_i.conj()  # (batch, num_pairs, time, stft_bins)

        # Phase transform (no-op if self.phase_transform is False)
        cps = self.apply_phase_transform(cps)

        # Naive SRP
        # DC_offset = stft.size(1) * stft.size(2)
        # srp = torch.zeros(size=(cps.size(0), cps.size(2), self.phase_shifts.size(1)), dtype=torch.float32,
        #                   device=cps.device)
        # for m in range(len(self.mic_pairs)):
        #     for direction in range(srp.size(2)):
        #         # only compute real part (imag part is normally discarded)
        #         srp[:, :, direction] += (cps[:, m].real * self.phase_shifts[m, direction].real
        #                                  - cps[:, m].imag * self.phase_shifts[m, direction].imag).sum(-1)
        # # Normalize SRP
        # srp = (2.0 * srp + DC_offset) / (len(self.frequencies) * len(self.mic_pairs))

        # Vectorized
        # Matmul multiplies the cps and phase shift corresponding to each frequency, and sums over frequencies,
        # for each steering vector
        # only compute real part (imag part is normally discarded)
        srp = (torch.matmul(cps.real, self.phase_shifts.real.transpose(-1, -2))
               - torch.matmul(cps.imag, self.phase_shifts.imag.transpose(-1, -2)))
        # Sum the srp for each direction over each microphone pair
        srp = srp.sum(dim=1)

        # Normalize srp
        #DC_offset = stft.size(1) * stft.size(2)
        #srp = (2.0 * srp + DC_offset) / (len(self.frequencies) * len(self.mic_pairs))

        # Output is (B, T, azimuth, elevation)
        return srp.reshape(srp.size(0), srp.size(1), len(self.azimuth_angles), len(self.elevation_angles))

    def estimate_doa(self, input_tensor: torch.Tensor):
        srp_phat = self(input_tensor)
        srp_phat = srp_phat.reshape(srp_phat.size(0), srp_phat.size(1), -1)
        max_indices = torch.argmax(srp_phat, dim=-1)
        return self.steering_vector_to_angle_map[max_indices]


if __name__ == '__main__':
    import copy

    import numpy as np
    import torchaudio


    def add_gaussian_noise(waveforms: torch.Tensor, snr_db: float) -> torch.Tensor:
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

    torch.manual_seed(0)

    # Sample rate and maximum delay in seconds
    sample_rate = 48000

    speech_signal, fs = torchaudio.load('/Users/JG96XG/PycharmProjects/aisnet/data/OSR_us_000_0010_8k.wav')
    if fs != sample_rate:
        speech_signal = torchaudio.functional.resample(speech_signal, fs, sample_rate)
    speech_signal = torch.randn_like(speech_signal)
    T = 1
    length = T * sample_rate
    speech_signal = speech_signal[:, :length]
    delay_samples_true = 5
    delay_time_true = delay_samples_true / sample_rate
    input_tensor = torch.stack([speech_signal, speech_signal, speech_signal, speech_signal], dim=0)
    if delay_samples_true > 0:
        left2 = copy.deepcopy(input_tensor).roll(-abs(delay_samples_true))
        left = input_tensor
        right = copy.deepcopy(input_tensor).roll(abs(delay_samples_true))
        right2 = copy.deepcopy(input_tensor).roll(2*abs(delay_samples_true))
    else:
        right = input_tensor
        right2 = copy.deepcopy(input_tensor).roll(-abs(delay_samples_true))
        left = copy.deepcopy(input_tensor).roll(abs(delay_samples_true))
        left2 = copy.deepcopy(input_tensor).roll(2 * abs(delay_samples_true))

    input_tensor = torch.concat([left, right], dim=1)
    input_tensor = add_gaussian_noise(input_tensor, snr_db=10)

    microphone_positions = torch.tensor([[0, 0.09, 0], [0, -0.09, 0]])
    dist_mics = torch.sqrt(torch.sum((microphone_positions[1] - microphone_positions[0]) ** 2))
    c_sound = 343
    doa_real = torch.arcsin(torch.tensor(delay_time_true * c_sound) / dist_mics) * (180. / torch.pi)

    # Instantiate the GCC-PHAT module
    elevation_range = (50., 140.)
    azimuth_range = (-90., 90.)
    frequency_range = (500., 4000.)
    srp_phat = SRP(microphone_positions=microphone_positions, n_fft=512, window_length=401, hop_length=160,
                   sample_rate=sample_rate, center=True,
                   c_sound=c_sound, elevation_range=elevation_range, azimuth_range=azimuth_range,
                   frequency_range=frequency_range)

    # Compute SRP-PHAT map
    srp_map = srp_phat(input_tensor)

    # Estimate doa
    doa = srp_phat.estimate_doa(input_tensor)

    # plot srp_map
    from matplotlib import pyplot as plt
    x = torch.arange(0., float(srp_map.size(1)))
    y = torch.rad2deg(srp_phat.azimuth_angles)
    plt.pcolormesh(x, y, srp_map[0, :, :, len(srp_phat.elevation_angles)//2].T, shading='auto')
    plt.xlabel(f"Time")
    plt.ylabel(f"Azimuth angle [degree]")
    plt.suptitle("SRP-PHAT estimation")
    plt.show()

    if srp_map.size(-1) > 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(torch.rad2deg(srp_phat.azimuth_angles), torch.rad2deg(srp_phat.elevation_angles))
        ax.plot_surface(X, Y, srp_map[0, 20, :, :].T, cmap='viridis')
        fig.show()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        X, Y = np.meshgrid(torch.rad2deg(srp_phat.azimuth_angles), torch.rad2deg(srp_phat.elevation_angles))
        ax2.pcolormesh(X, Y, srp_map[0, 20, :, :].T, cmap='viridis')
        fig2.show()
    print(f"Done")
