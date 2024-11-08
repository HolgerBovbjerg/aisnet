import torch
from torch import nn

from source.nnet.feature_extraction import GCCPHAT


def estimate_doa(tdoa: torch.Tensor, dist_mics: torch.Tensor, c_sound: float = 343.):
    dist_wave = c_sound * tdoa
    above = dist_wave >= dist_mics.unsqueeze(-1)
    below = dist_wave <= -dist_mics.unsqueeze(-1)
    dist_mics = dist_mics.unsqueeze(0).expand_as(above)
    dist_wave[above] = dist_mics[above] - 1.e-9
    dist_wave[below] = -dist_mics[below] + 1.e-9
    return torch.arcsin(dist_wave / dist_mics) * (180. / torch.pi)


class GCCPHATDoA(nn.Module):
    def __init__(self, n_fft: int, window_length: int, hop_length: int, sample_rate: int,
                 microphone_positions: torch.Tensor, center: bool = False,
                 window_type: str = "hann", c_sound=343.,
                 limit_to_max_delay: bool = False):
        super().__init__()
        assert microphone_positions.size(0) == 2, ("microphone positions must have size of (2, 3), "
                                                   "only 2 channel arrays are supported")
        self.sample_rate = sample_rate
        self.microphone_positions = microphone_positions
        self.n_mics = microphone_positions.shape[0]
        self.c_sound = c_sound
        self.mic_indices_i, self.mic_indices_j = torch.triu_indices(self.n_mics, self.n_mics, offset=1)
        self.mic_pairs = tuple((int(i), int(j)) for i, j in zip(self.mic_indices_i, self.mic_indices_j))
        self.dist_mics = self.get_mic_dists()
        self.max_delay = self.get_max_delay() if limit_to_max_delay else None
        self.gcc = GCCPHAT(n_fft=n_fft, window_length=window_length, hop_length=hop_length, sample_rate=sample_rate,
                           center=center, window_type=window_type, max_delay=self.max_delay, n_mics=self.n_mics)

    def get_mic_dists(self):
        # Compute pairwise distances between microphones
        dist_mics = torch.zeros(len(self.mic_pairs))
        for n, (i, j) in enumerate(self.mic_pairs):
            dist_mics[n] = torch.linalg.norm(self.microphone_positions[i] -self.microphone_positions[j])
        return dist_mics

    def get_max_delay(self):
        t_max = self.dist_mics / self.c_sound
        samples_max = torch.floor(t_max * self.sample_rate)
        return samples_max

    def forward(self, x):
        r = self.gcc(x)
        tdoa = self.gcc.estimate_tdoa(r)
        doa = estimate_doa(tdoa=tdoa[0], dist_mics=self.dist_mics, c_sound=self.c_sound)
        return doa.transpose(-1, -2)


if __name__ == "__main__":
    import copy
    # Microphone positions (assuming 3 microphones in a 2D plane, shape = (n_mics, 2))
    microphone_positions = torch.tensor([[0, 0.09, 0], [0, -0.09, 0]])

    # Example input tensor (batch, channels, time) = (2, 3, 1024)
    sample_rate = 16000
    delay_samples_true = 5
    input_tensor = torch.randn(4, 1, sample_rate)
    if delay_samples_true > 0:
        left = input_tensor
        right = copy.deepcopy(input_tensor).roll(abs(delay_samples_true))
    else:
        right = input_tensor
        left = copy.deepcopy(input_tensor).roll(abs(delay_samples_true))

    input_tensor = torch.cat([left, right], dim=1)

    delay_time_true = delay_samples_true / sample_rate

    dist_mics = torch.sqrt(torch.sum((microphone_positions[1] - microphone_positions[0]) ** 2))
    c_sound = 343
    doa_real = torch.arcsin(torch.tensor(delay_time_true * c_sound) / dist_mics) * (180. / torch.pi)
    resolution = torch.arcsin(torch.tensor((1 / sample_rate) * c_sound) / dist_mics) * (180. / torch.pi)

    # Create an instance of the GCCPHAT class
    gcc_phat_doa = GCCPHATDoA(n_fft=512, window_length=400, hop_length=160, sample_rate=sample_rate,
                              microphone_positions=microphone_positions, center=True, limit_to_max_delay=True)



    # Forward pass
    doa = gcc_phat_doa(input_tensor)
    print("Estimated DoA:", doa)
