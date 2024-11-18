from math import ceil

from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from source.nnet.feature_extraction import GCCPHAT, Gabor


def get_feature_extractor(name: str, **kwargs) -> nn.Module:
    if name == "gcc_phat":

        return GCCPHAT(**kwargs)
    else:
        return None


class MACSoundDoA(nn.Module):
    def __init__(self, num_filters, window_length, hop_length, microphone_positions: torch.Tensor,
                 sample_rate: int = 16000, c_sound: float = 343.,
                 elevation_resolution: float = 5., azimuth_resolution: float = 5.,
                 elevation_range: Tuple[float, float] = (0., 180.), azimuth_range: Tuple[float, float] = (-180., 180.),
                 use_time_delay: bool = False, causal=True, use_complex_convolution=True):
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
        avg_pool_size = 2
        last_avg_pool_size = num_filters // (avg_pool_size ** 4)
        assert last_avg_pool_size > 0, f"Minimum number of filters is {avg_pool_size ** 4}"
        self.window_length = window_length
        self.hop_length = hop_length
        # Create indices for the unique microphone pairs
        self.mic_indices_i, self.mic_indices_j = torch.triu_indices(self.n_mics, self.n_mics, offset=1)
        self.mic_pairs = tuple((int(i), int(j)) for i, j in zip(self.mic_indices_i, self.mic_indices_j))

        self.gabor = Gabor(num_filters=num_filters, n_coefficients=window_length, sample_rate=sample_rate,
                           causal=causal, use_complex_convolution=use_complex_convolution,
                           stride=hop_length)
        self.formant_enhancer = nn.Conv2d(num_filters, num_filters, kernel_size=(3, 1),
                                          padding="same")
        self.detail_enhancer = nn.Conv2d(num_filters, num_filters, kernel_size=(3, 1),
                                         padding="same")
        self.block_1 = nn.Sequential(*(nn.Conv2d(len(self.mic_indices_i) + self.n_mics, 64, kernel_size=3,
                                                 padding="same"),
                                       nn.Conv2d(64, 64, kernel_size=3,
                                                 padding="same"),
                                       nn.AvgPool2d(kernel_size=(avg_pool_size, avg_pool_size),
                                                    stride=(avg_pool_size, avg_pool_size))))
        self.block_2 = nn.Sequential(*(nn.Conv2d(64, 128, kernel_size=3,
                                                 padding="same"),
                                       nn.Conv2d(128, 128, kernel_size=3,
                                                 padding="same"),
                                       nn.AvgPool2d(kernel_size=(avg_pool_size, avg_pool_size),
                                                    stride=(avg_pool_size, avg_pool_size))))
        self.block_3 = nn.Sequential(*(nn.Conv2d(128, 256, kernel_size=3,
                                                 padding="same"),
                                       nn.Conv2d(256, 256, kernel_size=3,
                                                 padding="same"),
                                       nn.AvgPool2d(kernel_size=(avg_pool_size, 1), stride=(avg_pool_size, 1))))
        self.block_4 = nn.Sequential(*(nn.Conv2d(256, 512, kernel_size=3,
                                                 padding="same"),
                                       nn.Conv2d(512, 512, kernel_size=3,
                                                 padding="same"),
                                       nn.AvgPool2d(kernel_size=(avg_pool_size, 1), stride=(avg_pool_size, 1))))
        self.avg_pool = nn.AvgPool2d(kernel_size=(last_avg_pool_size, 1), stride=(last_avg_pool_size, 1)) \
            if last_avg_pool_size > 1 else nn.Identity()
        self.self_attention = nn.MultiheadAttention(512, 1)
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
            self.azimuth_classifier = nn.Linear(512 + self.n_mics * 3,
                                                len(self.azimuth_angles))
        if len(self.elevation_angles) > 1:
            self.elevation_classifier = nn.Linear(512 + self.n_mics * 3,
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
        x = self.gabor(x)
        x_gcc = (x[:, self.mic_indices_i] * x[:, self.mic_indices_j].conj()).real.permute(0, 2, 1, 3)
        x = torch.abs(x) ** 2
        x = x.permute(0, 2, 1, 3)
        x_formant = self.formant_enhancer(x)
        x_details = self.detail_enhancer(x)
        x = torch.cat((x_gcc, F.sigmoid(x_formant) + F.tanh(x_details)), dim=-2).transpose(1, 2)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.avg_pool(x).squeeze().transpose(1, 2)
        x, _ = self.self_attention(x, x, x)

        microphone_positions = self.microphone_positions.flatten().unsqueeze(0).unsqueeze(1)
        microphone_positions = microphone_positions.expand(x.size(0), x.size(1),
                                                           microphone_positions.size(-1))
        x = torch.cat((x, microphone_positions), dim=-1)

        if self.elevation_classifier:
            elevation_out = self.elevation_classifier(x)
            elevation_out = F.softmax(elevation_out, dim=-1)
            elevation_out = self.argmax(elevation_out, dim=-1)
        else:
            elevation_out = torch.ones(size=(x.size(0), x.size(1))) * self.elevation_angles[0]

        if self.azimuth_classifier:
            azimuth_out = self.azimuth_classifier(x)
            azimuth_out = F.softmax(azimuth_out, dim=-1)
            azimuth_out = torch.argmax(azimuth_out, dim=-1)
            azimuth_out = self.azimuth_angles[azimuth_out]
        else:
            azimuth_out = torch.ones(size=(x.size(0), x.size(1))) * self.azimuth_angles[0]

        return torch.stack([elevation_out, azimuth_out], dim=-1)


if __name__ == "__main__":
    import copy

    microphone_positions = torch.tensor([[0, -0.09, 0], [0, 0.09, 0], [0, 0.09, 0.09]])

    # Create an instance of the class
    elevation_range = (90., 90.)
    azimuth_range = (-90., 90.)
    macdoa = MACSoundDoA(num_filters=512, window_length=400, hop_length=160, sample_rate=16000,
                         microphone_positions=microphone_positions,
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
    doa = macdoa(input_tensor)
    print("Estimated DoA:", doa)
