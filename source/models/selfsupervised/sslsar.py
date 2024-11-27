from typing import Optional
from dataclasses import asdict

import torch
from torch import nn

from source.nnet.modules.input_projection import InputProjection


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
        self.spectral_feature_projection = InputProjection(self.feature_dim,
                                                  self.spectral_encoder_embedding_dim,
                                                  dropout_rate=feature_dropout,
                                                  dropout_first=feature_dropout_first) \
            if feature_projection else nn.Identity()
        self.spatial_feature_projection = InputProjection(self.feature_dim,
                                                           self.spatial_encoder_embedding_dim,
                                                           dropout_rate=feature_dropout,
                                                           dropout_first=feature_dropout_first) \
            if feature_projection else nn.Identity()
        self.spectral_encoder = spectral_encoder
        self.spatial_encoder = spatial_encoder
        self.mask_generator = mask_generator
        self.spectral_mask_embedding = nn.Parameter(
            torch.FloatTensor(self.spectral_encoder_embedding_dim).uniform_()
        )
        self.spatial_mask_embedding = nn.Parameter(
            torch.FloatTensor(self.spatial_encoder_embedding_dim).uniform_()
        )
        self.decoder = decoder if decoder is not None \
            else nn.Conv1d(in_channels=2*self.spectral_encoder_embedding_dim + 2*self.spectral_encoder_embedding_dim,
                           out_channels=self.feature_dim * n_feature_channels,
                           kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, target: Optional[torch.Tensor] = None):
        # Initial length of input
        init_length = x.size(-1)

        # Extract input and target features
        x = self.feature_extractor(x)
        if target is None:
            target = x.clone().detach()
        else:
            # Create GCC target
            target = self.feature_extractor(target)

        # Compute new lengths after feature extraction
        extra = init_length % x.size(-2)
        if extra > 0:
            lengths = lengths - extra
        lengths = lengths // (init_length // x.size(-2))

        # Project features to input size of encoder if feature_projection was specified in module creation, else no-op.
        x_spatial = x
        x_spectral = x.clone()
        x_spectral = self.spectral_feature_projection(x_spectral)
        x_spatial = self.spatial_feature_projection(x_spatial)

        # Masking
        mask = self.mask_generator.generate_mask(x_spatial[:, 0], lengths).unsqueeze(1)
        spatial_mask = torch.concat([mask, mask], dim=1)
        spectral_mask = spatial_mask.clone()
        spectral_mask[:, 1] = ~spectral_mask[:, 1]
        x_spatial[spatial_mask] = self.spatial_mask_embedding.to(x_spatial.dtype)
        x_spectral[spectral_mask] = self.spectral_mask_embedding.to(x_spatial.dtype)

        # Stack features for each channel if features are computed channel-wise
        x_spectral = x_spectral.transpose(1, 2)
        x_spectral = x_spectral.reshape(x_spectral.size(0), x_spectral.size(1), -1)
        x_spatial = x_spatial.transpose(1, 2)
        x_spatial = x_spatial.reshape(x_spatial.size(0), x_spatial.size(1), -1)

        # Encode features
        x_spectral, _ = self.spectral_encoder(x_spectral, lengths)
        x_spatial, _ = self.spectral_encoder(x_spatial, lengths)

        # Concatenate spectral and spatial features
        x = torch.concat([x_spectral, x_spatial], dim=-1)

        # Put encoded features through decoder to predict target
        x = self.decoder(x.transpose(-1, -2)).reshape(x.size(0), self.n_feature_channels, x.size(1), -1)


        x_spectral = x[spectral_mask]
        x_spatial = x[spatial_mask]
        target_spectral = target[spectral_mask]
        target_spatial = target[spatial_mask]

        x = torch.concat([x_spectral, x_spatial], dim=0)
        target = torch.concat([target_spectral, target_spatial], dim=0)

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
