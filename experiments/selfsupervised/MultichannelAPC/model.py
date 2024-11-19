from dataclasses import dataclass, field
from logging import getLogger

from omegaconf import OmegaConf
import torch
from torch import nn


from source.utils import count_parameters
from source.nnet.feature_extraction import build_extractor, FeatureExtractorConfig
from source.nnet.encoder import build_encoder, EncoderConfig
from source.models.selfsupervised.MultichannelAPC import MultichannelAPC
from source.nnet.utils.padding import lengths_to_padding_mask


logger = getLogger(__name__)


@dataclass
class ModelConfig:
    input_dim: int = 40
    output_dim: int = 40
    feature_projection: bool = True
    feature_dropout: float = 0.
    encoder_embedding_dim: int = 512
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    sample_rate: int = 16000
    n_channels: int = 2


class Model(nn.Module):
    """
    Main module.
    """

    def __init__(self,
                 cfg: ModelConfig = ModelConfig()):
        super().__init__()
        feature_projection = nn.Linear(cfg.input_dim, cfg.encoder_embedding_dim) if cfg.feature_projection \
            else nn.Identity()
        feature_extractor = build_extractor(cfg.feature_extractor)
        encoder =  build_encoder(cfg.encoder)
        self.model = MultichannelAPC(input_dim=cfg.input_dim,
                                     feature_extractor=feature_extractor,
                                     feature_dropout=cfg.feature_dropout,
                                     feature_projection=feature_projection,
                                     encoder=encoder,
                                     encoder_embedding_dim=cfg.encoder_embedding_dim,
                                     n_channels=cfg.n_channels)

    def forward(self, x, lengths, target):
        prediction, target, lengths = self.model(x, lengths=lengths, target=target)
        return prediction, target, lengths


def build_model(config):
    logger.info(f"Model Config: {OmegaConf.to_object(config.model)}")
    model_config = ModelConfig(**config.model)
    model = Model(model_config)
    logger.info(f"Created model with {count_parameters(model)} parameters.")
    return model
