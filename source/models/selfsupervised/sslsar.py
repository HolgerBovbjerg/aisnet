from typing import Optional
from dataclasses import asdict

import torch
from torch import nn

from source.nnet.modules.input_projection import InputProjection
from source.nnet.utils.masking import lengths_to_padding_mask


class SSLSARDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim: int = 3072):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SSLSAR(nn.Module):
    """
    SSLSAR main module.
    """

    def __init__(self,
                 feature_extractor: nn.Module,
                 feature_dim: int,
                 spectral_encoder: nn.Module,
                 spatial_encoder: nn.Module,
                 spectral_encoder_embedding_dim: int,
                 spatial_encoder_embedding_dim: int,
                 mask_generator: nn.Module,
                 feature_projection: bool = True,
                 feature_dropout: float = 0.,
                 feature_dropout_first: bool = False,
                 decoder: Optional[nn.Module] = None,
                 n_feature_channels: int = 2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        self.mask_generator = mask_generator
        self.spectral_encoder_embedding_dim = spectral_encoder_embedding_dim
        self.spatial_encoder_embedding_dim = spatial_encoder_embedding_dim
        self.n_feature_channels = n_feature_channels
        self.spectral_feature_projection = InputProjection(2*self.feature_dim,
                                                  self.spectral_encoder_embedding_dim,
                                                  dropout_rate=feature_dropout,
                                                  dropout_first=feature_dropout_first) \
            if feature_projection else nn.Identity()
        self.spatial_feature_projection = InputProjection(2*self.feature_dim,
                                                           self.spatial_encoder_embedding_dim,
                                                           dropout_rate=feature_dropout,
                                                           dropout_first=feature_dropout_first) \
            if feature_projection else nn.Identity()
        self.spectral_encoder = spectral_encoder
        self.spatial_encoder = spatial_encoder
        self.mask_generator = mask_generator
        self.mask_embedding = nn.Parameter(
            torch.FloatTensor(self.feature_dim).uniform_()
        )
        self.decoder = decoder if decoder is not None \
            else SSLSARDecoder(input_size=self.spatial_encoder_embedding_dim + self.spectral_encoder_embedding_dim,
                               output_size=self.feature_dim * self.n_feature_channels,
                               hidden_dim=3072)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, target: Optional[torch.Tensor] = None):
        # Initial length of input
        init_length = x.size(-1)

        # Extract input and target features
        x = self.feature_extractor(x)
        if target is None:
            target = x.clone().detach()
        else:
            # Create target
            target = self.feature_extractor(target)

        # Compute new lengths after feature extraction
        extra = init_length % x.size(-2)
        if extra > 0:
            lengths = lengths - extra
        lengths = lengths // (init_length // x.size(-2))

        # Project features to input size of encoder if feature_projection was specified in module creation, else no-op.
        x_spatial = x
        x_spectral = x.clone()

        # Masking
        mask = self.mask_generator.generate_mask(x_spatial[:, 0], lengths).unsqueeze(1)
        spatial_mask = torch.concat([mask, mask], dim=1)
        # Clone spatial mask
        spectral_mask = spatial_mask.clone()
        # Randomly choose channel 0 or channel 1 as target
        target_channel_0 = torch.rand(spectral_mask.size(0), device=spectral_mask.device) > 0.5
        # Random invert spectral mask for channel which is not target
        spectral_mask[target_channel_0, 0] = ~spectral_mask[target_channel_0, 0]
        spectral_mask[~target_channel_0, 1] = ~spectral_mask[~target_channel_0, 1]
        # Ensure padding is not masked
        padding_mask = lengths_to_padding_mask(lengths).unsqueeze(1).repeat(1, 2, 1)
        spatial_mask[padding_mask] = False
        spectral_mask[padding_mask] = False
        # Apply mask
        x_spatial[spatial_mask] = self.mask_embedding.to(x_spatial.dtype)
        x_spectral[spectral_mask] = self.mask_embedding.to(x_spectral.dtype)

        # Stack features for each channel if features are computed channel-wise
        x_spectral = x_spectral.transpose(1, 2)
        x_spectral = x_spectral.reshape(x_spectral.size(0), x_spectral.size(1), -1)
        x_spatial = x_spatial.transpose(1, 2)
        x_spatial = x_spatial.reshape(x_spatial.size(0), x_spatial.size(1), -1)

        # Project features to encoder input size
        x_spectral = self.spectral_feature_projection(x_spectral)
        x_spatial = self.spatial_feature_projection(x_spatial)

        # Encode features
        x_spectral, _ = self.spectral_encoder(x_spectral, lengths)
        x_spatial, _ = self.spatial_encoder(x_spatial, lengths)

        # Concatenate spectral and spatial features
        x = torch.concat([x_spectral, x_spatial], dim=-1)

        # Put encoded features through decoder to predict target and reshape to have prediction for each channel
        x = self.decoder(x).reshape(x.size(0), x.size(1), self.n_feature_channels, -1).transpose(1, 2)

        # Generate prediction and target (randomly selected channel)
        x = x[torch.stack([target_channel_0, ~target_channel_0], dim=-1)]
        target = target[torch.stack([target_channel_0, ~target_channel_0], dim=-1)]

        # Select only masked part
        x = x[mask[:, 0]]
        target = target[mask[:, 0]]

        return x, target, lengths

if __name__ == "__main__":
    from source.nnet.feature_extraction import STFT, STFTConfig
    from source.nnet.encoder import LSTMEncoder, LSTMEncoderConfig
    from source.nnet.utils.masking import MaskGeneratorConfig, MaskGenerator

    input_signal = torch.randn(size=(10, 2, 10000))

    feature_extractor = STFT(**asdict(STFTConfig(n_fft=512, window_length=400, hop_length=160, output_type="log_power_phase")))

    encoder = LSTMEncoder(**asdict(LSTMEncoderConfig(input_dim=128, hidden_dim=64, num_layers=1)))

    model = SSLSAR(feature_dim=514,
                   feature_extractor=feature_extractor,
                   spectral_encoder=encoder,
                   spectral_encoder_embedding_dim=64,
                   spatial_encoder=encoder,
                   spatial_encoder_embedding_dim=64,
                   n_feature_channels=2,
                   mask_generator=MaskGenerator(MaskGeneratorConfig()))

    out = model(input_signal,
                lengths=torch.tensor([signal.size(-1) for signal in input_signal]),
                target=input_signal + torch.randn_like(input_signal))

    print(out)
