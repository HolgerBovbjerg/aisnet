from typing import Optional
import copy

import torch
from torch import nn


class MultichannelAPC(nn.Module):
    """
    Denoising Autoregressive Predictive Coding main module.
    """

    def __init__(self,
                 feature_extractor: nn.Module,
                 input_dim: int,
                 encoder: nn.Module,
                 encoder_embedding_dim: int,
                 feature_projection: nn.Module = nn.Identity(),
                 feature_dropout: float = 0.,
                 decoder: Optional[nn.Module] = None,
                 n_channels: Optional[int] = None,
                 output_dim: Optional[int] = None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.input_dim = input_dim
        self.feature_projection = feature_projection
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.encoder = encoder
        self.encoder_embedding_dim = encoder_embedding_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.decoder = decoder if decoder is not None \
            else nn.Conv1d(in_channels=encoder_embedding_dim,
                           out_channels=self.output_dim,
                           kernel_size=1, stride=1)

    def forward(self, x, lengths, target: Optional[torch.Tensor], time_shift: int = 3, extract_features: bool = False):
        # Initial length of input
        init_length = x.size(-1)

        # Extract input and target features
        x = self.feature_extractor(x).transpose(-1, -2)
        x = self.feature_projection(x)
        # Compute target. If target is not provided, x is used as the target
        if target is not None:
            target = self.feature_extractor(target).transpose(-1, -2)
        else:
            target = copy.deepcopy(x)

        # Compute new lengths after feature extraction
        extra = init_length % x.size(-2)
        if extra > 0:
            lengths = lengths - extra
        lengths = lengths // (init_length // x.size(-2))

        # Dropout features
        x = self.feature_dropout(x)

        # Stack features for each channel if features are computed channelwise
        if len(x.size()) == 4:
            x = x.transpose(1, 2)
            x = x.reshape(x.size(0), x.size(1), -1)
            target = target.transpose(1, 2)
            target = target.reshape(x.size(0), x.size(1), -1)

        # Create time shifted input and target
        x = x[:, :-time_shift]
        target = target[:, time_shift:]

        # After time shift lengths are 'time_shift' shorter
        lengths = lengths - time_shift

        # Encode features
        x, hidden = self.encoder(x, lengths)

        # If only features are needed return them
        if extract_features:
            return x
        # Else put encoded features through post_network to predict target
        x = self.decoder(x.transpose(-1, -2)).transpose(-1, -2)
        return x, target, lengths
