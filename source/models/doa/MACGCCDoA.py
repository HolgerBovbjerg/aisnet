from math import ceil

from typing import Tuple
import torch
from torch import nn

from source.nnet.feature_extraction import GCC


def get_feature_extractor(name: str, **kwargs) -> nn.Module:
    if name == "gcc_phat":

        return GCC(**kwargs)
    else:
        return None


class MACGCCDoA(nn.Module):
    def __init__(self, n_fft, window_length, hop_length, microphone_positions: torch.Tensor, sample_rate: int = 16000,
                 center: bool = False, window_type: str = 'hann', c_sound: float = 343.,
                 elevation_resolution: float = 5., azimuth_resolution: float = 5.,
                 elevation_range: Tuple[float, float] = (0., 180.), azimuth_range: Tuple[float, float] = (-180., 180.),
                 use_time_delay: bool = False):
        super().__init__()
        self.sample_rate = sample_rate
        self.microphone_positions = microphone_positions
        self.n_mics = microphone_positions.size(0)
        self.c_sound = c_sound
        self.use_time_delay = use_time_delay
        self.mic_indices_i, self.mic_indices_j = torch.triu_indices(self.n_mics, self.n_mics, offset=1)
        self.mic_pairs = tuple((int(i), int(j)) for i, j in zip(self.mic_indices_i, self.mic_indices_j))
        diff = microphone_positions.unsqueeze(1) - microphone_positions.unsqueeze(0)
        dist_mics = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        dist_mics = dist_mics[self.mic_indices_i, self.mic_indices_j]
        c_sound = 343
        t_max = dist_mics / c_sound
        t_max_samples = torch.floor(t_max * sample_rate)
        self.gcc_phat = GCC(n_fft=n_fft, window_length=window_length, hop_length=hop_length,
                                sample_rate=sample_rate, window_type=window_type,
                                center=center, max_delay=t_max_samples, n_mics=microphone_positions.size(0))
        self.elevation_resolution = elevation_resolution
        self.elevation_range = elevation_range
        self.azimuth_resolution = azimuth_resolution
        self.azimuth_range = azimuth_range
        self.azimuth_angles = None
        self.elevation_angles = None
        self.elevation_angles, self.azimuth_angles = self._get_elevation_and_azimuth_angles()
        self.azimuth_classifier = None
        self.elevation_classifier = None
        if len(self.azimuth_angles) > 1:
            if self.use_time_delay:
                self.azimuth_classifier = nn.Linear(len(self.mic_pairs) + self.microphone_positions.flatten().size(0),
                                                    len(self.azimuth_angles))
            else:
                self.azimuth_classifier = nn.Linear(int(2 * torch.sum(t_max_samples).item())
                                                    + self.microphone_positions.flatten().size(0),
                                                    len(self.azimuth_angles))
        if len(self.elevation_angles) > 1:
            if self.use_time_delay:
                self.elevation_classifier = nn.Linear(len(self.mic_pairs) + self.microphone_positions.flatten().size(0),
                                                      len(self.elevation_angles))
            else:
                self.elevation_classifier = nn.Linear(int(2 * torch.sum(t_max_samples).item())
                                                      + self.microphone_positions.flatten().size(0),
                                                      len(self.elevation_angles))

    def _get_elevation_and_azimuth_angles(self):
        elevation_span = abs(self.elevation_range[1] - self.elevation_range[0])
        num_directions_elevation = ceil(elevation_span / self.elevation_resolution) if elevation_span else 1
        if elevation_span:
            elevation_angles = torch.arange(num_directions_elevation) * self.elevation_resolution + \
                               self.elevation_range[0]
        else:
            elevation_angles = torch.tensor([self.elevation_range[0]])
        azimuth_span = abs(self.azimuth_range[1] - self.azimuth_range[0])
        num_directions_azimuth = ceil(azimuth_span / self.azimuth_resolution) if azimuth_span else 1
        if azimuth_span:
            azimuth_angles = torch.arange(num_directions_azimuth) * self.azimuth_resolution + self.azimuth_range[0]
        else:
            azimuth_angles = torch.tensor([self.azimuth_range[0]])

        return elevation_angles, azimuth_angles

    def forward(self, x):
        x = self.gcc_phat(x)
        if self.use_time_delay:
            x = self.gcc_phat.estimate_tdoa(x)

        x = torch.stack(x, dim=-1)

        x = x.reshape(x.size(0), x.size(1), -1)

        microphone_positions = self.microphone_positions.flatten().unsqueeze(0).unsqueeze(1)
        microphone_positions = microphone_positions.expand(x.size(0), x.size(1),
                                                           microphone_positions.size(-1))
        x = torch.cat((x, microphone_positions), dim=-1)

        if self.elevation_classifier:
            elevation_out = self.elevation_classifier(x)
        else:
            elevation_out = torch.ones(x.size(0), x.size(1), 1) * self.elevation_angles[0]

        if self.azimuth_classifier:
            azimuth_out = self.azimuth_classifier(x)
        else:
            azimuth_out = torch.ones(x.size(0), x.size(1), 1) * self.azimuth_angles[0]

        return elevation_out, azimuth_out


if __name__ == "__main__":
    import copy

    microphone_positions = torch.tensor([[0, -0.09, 0], [0, 0.09, 0], [0, 0.09, 0.09]])

    # Create an instance of the class
    elevation_range = (90., 90.)
    azimuth_range = (-90., 90.)
    macgccdoa = MACGCCDoA(n_fft=512, window_length=400, hop_length=160, sample_rate=16000,
                          microphone_positions=microphone_positions, center=True,
                          azimuth_range=azimuth_range, azimuth_resolution=5.,
                          elevation_range=elevation_range, elevation_resolution=5.,
                          use_time_delay=True)

    # Example input tensor
    sample_rate = 16000
    c_sound = 343.
    dist_mics = torch.sqrt(torch.sum((microphone_positions[1] - microphone_positions[0]) ** 2))
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
    input_tensor = torch.cat([left, right, right], dim=1)

    # Forward pass
    doa = macgccdoa(input_tensor)
    doa_elevation = doa[0]
    doa_azimuth = macgccdoa.azimuth_angles[doa[1].argmax(dim=-1)]
    print("Estimated DoA:", doa)
