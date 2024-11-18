from typing import Tuple
import torch
from torch import nn

from source.nnet.feature_extraction import SRP


class SRPPHATDoA(nn.Module):
    def __init__(self, n_fft, window_length, hop_length, microphone_positions: torch.Tensor, sample_rate: int = 16000,
                 center: bool = False, window_type: str = 'hann',
                 elevation_resolution: float = 5., azimuth_resolution: float = 5.,
                 elevation_range: Tuple[float, float] = (0., 180.), azimuth_range: Tuple[float, float] = (-180., 180.),
                 c_sound: float = 343., frequency_range: Tuple[float, float] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.microphone_positions = microphone_positions
        self.n_mics = microphone_positions.size(0)
        self.c_sound = c_sound
        self.srp = SRP(microphone_positions=microphone_positions, n_fft=n_fft, window_length=window_length,
                       hop_length=hop_length, sample_rate=sample_rate,
                       center=center, c_sound=c_sound, window_type=window_type, elevation_range=elevation_range,
                       azimuth_range=azimuth_range, elevation_resolution=elevation_resolution,
                       azimuth_resolution=azimuth_resolution, frequency_range=frequency_range)

    def forward(self, x):
        doa = self.srp.estimate_doa(x)
        return doa


if __name__ == "__main__":
    import copy

    microphone_positions = torch.tensor([[0, -0.09, 0], [0, 0.09, 0]])

    # Create an instance of the class
    elevation_range = (90., 90.)
    azimuth_range = (-90., 90.)
    frequency_range = (500., 4000.)
    srp_phat = SRPPHATDoA(n_fft=512, window_length=400, hop_length=160, sample_rate=16000,
                          microphone_positions=microphone_positions, center=True,
                          elevation_range=elevation_range, azimuth_range=azimuth_range,
                          frequency_range=frequency_range)

    # Example input tensor
    sample_rate = 16000
    c_sound = 343.
    dist_mics = torch.sqrt(torch.sum((microphone_positions[1] - microphone_positions[0])**2))
    delay_samples_true = 2
    delay_time_true = delay_samples_true / sample_rate
    doa_real = torch.arcsin(torch.tensor(delay_time_true * c_sound) / dist_mics) * (180. / torch.pi)
    torch.manual_seed(0)
    input_tensor = torch.randn(2, 1, 16000)
    if delay_samples_true < 0:
        left = input_tensor
        right = copy.deepcopy(input_tensor).roll(abs(delay_samples_true))
    else:
        right = input_tensor
        left = copy.deepcopy(input_tensor).roll(abs(delay_samples_true))
    input_tensor = torch.cat([left, right], dim=1)

    # Forward pass
    doa = srp_phat(input_tensor)
    print("Estimated DoA:", doa)
