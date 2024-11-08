from typing import Optional

import torch
from torchaudio.transforms import TimeStretch, TimeMasking, FrequencyMasking


class SpecAugment(torch.nn.Module):
    def __init__(self, time_mask_max_length: int = 30, p_max: float = 1., n_time_mask: int = 10,
                 freq_mask_max_length: int = 50, n_freq_mask: int = 2,
                 use_time_stretch: bool = False, rate_min: Optional[float] = 0.9, rate_max: Optional[float] = 1.2,
                 freq_bins: Optional[int] = 512):
        super().__init__()
        if use_time_stretch:
            self.stretch = TimeStretch(n_freq=freq_bins)
            self.rate_min = rate_min
            self.rate_max = rate_max
        self.time_mask = TimeMasking(time_mask_param=time_mask_max_length, p=p_max)
        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_max_length)

    def forward(self, x):
        x = x.transpose(-1, -2).unsqueeze(0)
        # TODO: Make time stretch work for SpecAugment
        # x = self.stretch(x, overriding_rate=random.uniform(self.rate_min, self.rate_max))
        for _ in range(self.n_time_mask):
            x = self.time_mask(x)
        for _ in range(self.n_freq_mask):
            x = self.freq_mask(x)
        x = x.transpose(-1, -2).squeeze()
        return x