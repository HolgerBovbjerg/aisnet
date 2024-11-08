from math import ceil

from typing import Tuple
import torch
from torch import nn


class DoANet(nn.Module):
    def __init__(self, feature_extractor, encoder, microphone_positions: torch.Tensor, sample_rate: int = 16000,
                 hidden_dim: int = 512,
                 elevation_resolution: float = 5., azimuth_resolution: float = 5.,
                 elevation_range: Tuple[float, float] = (0., 180.),
                 azimuth_range: Tuple[float, float] = (-180., 180.),
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.microphone_positions = microphone_positions
        self.n_mics = microphone_positions.size(0)
        self.elevation_resolution = elevation_resolution
        self.elevation_range = elevation_range
        self.azimuth_resolution = azimuth_resolution
        self.azimuth_range = azimuth_range
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.elevation_angles, self.azimuth_angles = self._get_elevation_and_azimuth_angles()
        if len(self.azimuth_angles) > 1:
            self.azimuth_classifier = nn.Linear(hidden_dim,
                                                len(self.azimuth_angles))
        else:
            self.azimuth_classifier = None
        if len(self.elevation_angles) > 1:
            self.elevation_classifier = nn.Linear(hidden_dim,
                                                  len(self.elevation_angles))
        else:
            self.elevation_classifier = None

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
        x = self.feature_extractor(x)
        x = self.encoder(x.reshape(x.size(0), x.size(1), -1))
        if self.elevation_classifier:
            elevation_out = self.elevation_classifier(x)
        else:
            elevation_out = torch.ones(x.size(0), x.size(1), 1) * self.elevation_angles[0]

        if self.azimuth_classifier:
            azimuth_out = self.azimuth_classifier(x)
        else:
            azimuth_out = torch.ones(x.size(0), x.size(1), 1) * self.azimuth_angles[0]
        return elevation_out, azimuth_out


if __name__ == '__main__':
    from models.feature_extraction import SRPPHAT

    B = 10
    T = 16000 * 4
    n_mics = 3
    microphone_positions = torch.tensor([[0, -0.09, 0], [0, 0.09, 0], [0, 0.09, 0.09]])
    input_tensor = torch.randn((B, 1, T))
    input_tensor = torch.cat([input_tensor.roll(i * 5) for i in range(n_mics)], dim=1)

    # Create an instance of the class
    elevation_range = (90., 90.)
    elevation_resolution = 5.
    azimuth_range = (-90., 90.)
    azimuth_resolution = 5.

    feature_extractor = SRPPHAT(microphone_positions=microphone_positions,
                                elevation_range=elevation_range, elevation_resolution=elevation_resolution,
                                azimuth_resolution=azimuth_resolution, azimuth_range=azimuth_range)

    encoder = nn.Linear(len(feature_extractor.azimuth_angles)*len(feature_extractor.elevation_angles), 512)

    doanet = DoANet(feature_extractor=feature_extractor,
                    encoder=encoder,
                    microphone_positions=microphone_positions,
                    sample_rate=16000,
                    elevation_range=elevation_range,
                    elevation_resolution=elevation_resolution,
                    azimuth_range=azimuth_range,
                    azimuth_resolution=azimuth_resolution)

    out = doanet(input_tensor)
    print("done")

