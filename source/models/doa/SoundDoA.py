import torch
from torch import nn
from torch.nn import functional as F

from source.nnet.feature_extraction import Gabor


class SoundDoA(nn.Module):
    def __init__(self, num_filters=256, window_length=1024, hop_length=600, sample_rate=16000, causal=True,
                 use_complex_convolution=True, num_classes: int = 14, n_mics: int = 4):
        super().__init__()
        avg_pool_size = 2
        last_avg_pool_size = num_filters // (avg_pool_size ** 4)
        assert last_avg_pool_size > 0, f"Minimum number of filters is {avg_pool_size ** 4}"
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_mics = n_mics
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
        self.block_1 = nn.Sequential(*(nn.Conv2d(len(self.mic_indices_i) + n_mics, 64, kernel_size=3,
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
        self.fc_semantic = nn.Linear(512, num_classes)
        self.fc_location = nn.Linear(512, num_classes * 3)

    def forward(self, x):
        x = self.gabor(x)
        x_gcc = (x[:, self.mic_indices_i] * x[:, self.mic_indices_j].conj()).real.permute(0, 2, 1, 3)
        x = torch.abs(x)**2
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
        semantic_category = self.fc_semantic(x)
        location = self.fc_location(x).reshape(x.size(0), x.size(1), -1, 3)
        return semantic_category, location


if __name__ == '__main__':
    from source.utils import count_parameters

    B = 10
    T = 16000*4
    n_mics = 2
    num_filters = 40
    input_tensor = torch.randn((B, 1, T))
    input_tensor = torch.cat([input_tensor.roll(i*5) for i in range(n_mics)], dim=1)
    model = SoundDoA(num_filters=num_filters, window_length=400, hop_length=160, n_mics=n_mics)
    out = model(input_tensor)

    print(f"Model has: {count_parameters(model)} parameters")
    print("done")
