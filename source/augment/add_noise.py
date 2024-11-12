from typing import Union

import torch
from torch_audiomentations import AddBackgroundNoise


class AddNoise(torch.nn.Module):
    def __init__(self, noise_paths: Union[str, list], sampling_rate: int = 16000, snr_db_min: int = 3,
                 snr_db_max: int = 30, p: float = 1.):
        super().__init__()
        self.noise_paths = noise_paths
        self.sampling_rate = sampling_rate
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.p = p
        self.add_noise = AddBackgroundNoise(background_paths=noise_paths,
                                            sample_rate=self.sampling_rate,
                                            min_snr_in_db=self.snr_db_min,
                                            max_snr_in_db=self.snr_db_max,
                                            p=self.p,
                                            output_type="dict")

    def forward(self, x: torch.Tensor):
        return self.add_noise(x).samples

