from typing import Optional
import copy
from dataclasses import asdict

import torch
from torch import nn

from source.nnet.feature_extraction import GCC, GCCConfig
from source.nnet.modules.input_projection import InputProjection


class NeuralGCC(nn.Module):
    """
    NeuralGCC main module.
    """

    def __init__(self,
                 feature_extractor: nn.Module,
                 feature_dim: int,
                 encoder: nn.Module,
                 encoder_embedding_dim: int,
                 gcc_dim: int,
                 gcc_config: GCCConfig,
                 feature_projection: bool = True,
                 feature_dropout: float = 0.,
                 feature_dropout_first: bool = False,
                 decoder: Optional[nn.Module] = None,
                 n_feature_channels: Optional[int] = 2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.n_feature_channels = n_feature_channels
        self.feature_projection = InputProjection(self.feature_dim * self.n_feature_channels,
                                                  self.encoder_embedding_dim,
                                                  dropout_rate=feature_dropout,
                                                  dropout_first=feature_dropout_first) \
            if feature_projection else nn.Identity()
        self.encoder = encoder
        self.gcc = GCC(**asdict(gcc_config))
        self.gcc_dim = gcc_dim
        self.decoder = decoder if decoder is not None \
            else nn.Conv1d(in_channels=encoder_embedding_dim,
                           out_channels=self.gcc_dim,
                           kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, target: Optional[torch.Tensor] = None,
                time_shift: int = 3, extract_features: bool = False):
        # Initial length of input
        init_length = x.size(-1)

        if target is None:
            target = self.gcc(x)
        else:
            # Create GCC target
            target = self.gcc(target)
        target = target[..., time_shift:]

        # Extract input and target features
        x = self.feature_extractor(x)

        # Stack features for each channel if features are computed channel-wise
        if len(x.size()) == 4:
            x = x.transpose(1, 2)
            x = x.reshape(x.size(0), x.size(1), -1)

        # Project features to input size of encoder if feature_projection was specified in module creation, else no-op.
        x = self.feature_projection(x)

        # Compute new lengths after feature extraction
        extra = init_length % x.size(-2)
        if extra > 0:
            lengths = lengths - extra
        lengths = lengths // (init_length // x.size(-2))

        # Create time shifted input and target
        x = x[:, :-time_shift]
        lengths = lengths - time_shift

        # Encode features
        x, hidden = self.encoder(x, lengths)

        # If only features are needed return them
        if extract_features:
            return x
        # Else put encoded features through post_network to predict target
        x = self.decoder(x.transpose(-1, -2))

        return x, target, lengths

if __name__ == "__main__":
    from source.nnet.feature_extraction import STFT, STFTConfig
    from source.nnet.encoder import LSTMEncoder, LSTMEncoderConfig

    input_signal = torch.randn(size=(10, 2, 10000))

    feature_extractor = STFT(**asdict(STFTConfig(n_fft=512, window_length=400, hop_length=160, output_type="log_power_phase")))

    encoder = LSTMEncoder(**asdict(LSTMEncoderConfig(input_dim=64, hidden_dim=64, num_layers=1)))

    max_delay = 8

    gcc_config = GCCConfig(max_delay=torch.tensor([max_delay,]), center=True, n_fft=512, window_length=400, hop_length=160)


    model = NeuralGCC(feature_dim=514,
                      feature_extractor=feature_extractor,
                      encoder=encoder,
                      encoder_embedding_dim=64,
                      gcc_config=gcc_config,
                      gcc_dim=max_delay*2+1,
                      n_feature_channels=2)

    out = model(input_signal,
                lengths=torch.tensor([signal.size(-1) for signal in input_signal]),
                target=input_signal + torch.randn_like(input_signal))

    print(out)