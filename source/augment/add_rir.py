from typing import Union

import torch
from torch_audiomentations import ApplyImpulseResponse


class AddRIR(torch.nn.Module):
    def __init__(self, rir_paths: Union[str, list], sampling_rate: int = 16000, p=1., convolve_mode="full",
                 compensate_for_propagation_delay=False, mode="per_example"):
        super().__init__()
        self.rir_paths = rir_paths
        self.sampling_rate = sampling_rate
        self.p = p
        self.convolve_mode = convolve_mode
        self.compensate_for_propagation_delay = compensate_for_propagation_delay
        self.mode = mode
        self.apply_rir = ApplyImpulseResponse(ir_paths=self.rir_paths,
                                              convolve_mode=self.convolve_mode,
                                              compensate_for_propagation_delay=self.compensate_for_propagation_delay,
                                              mode=self.mode,
                                              p=self.p,
                                              sample_rate=self.sampling_rate)

    def forward(self, x):
        return self.apply_rir(x)
