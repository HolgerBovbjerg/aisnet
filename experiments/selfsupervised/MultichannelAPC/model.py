from dataclasses import dataclass
from logging import getLogger
from typing import Union

from omegaconf import OmegaConf, DictConfig
from torch import nn

from source.utils import count_parameters
from source.nnet.feature_extraction import build_feature_extractor, FeatureExtractorConfig
from source.nnet.encoder import build_encoder, EncoderConfig
from source.models.selfsupervised.MultichannelAPC import MultichannelAPC


logger = getLogger(__name__)


@dataclass
class ModelConfig:
    feature_extractor: Union[FeatureExtractorConfig, dict]
    encoder: Union[EncoderConfig, dict]
    input_dim: int = 40
    output_dim: int = 40
    feature_projection: bool = True
    feature_dropout: float = 0.
    encoder_embedding_dim: int = 512
    sample_rate: int = 16000


    def __post_init__(self):
        if isinstance(self.encoder, (dict, DictConfig)):
            self.encoder = EncoderConfig(**self.encoder)
        elif not isinstance(self.encoder, EncoderConfig):
            raise ValueError("Wrong input for 'encoder'.")
        if isinstance(self.feature_extractor, (dict, DictConfig)):
            self.feature_extractor = FeatureExtractorConfig(**self.feature_extractor)
        elif not isinstance(self.feature_extractor, FeatureExtractorConfig):
            raise ValueError("Wrong input for 'feature_extractor'.")


class Model(nn.Module):
    """
    Main module.
    """

    def __init__(self,
                 cfg: ModelConfig = ModelConfig()):
        super().__init__()
        feature_projection = nn.Linear(cfg.input_dim, cfg.encoder_embedding_dim) if cfg.feature_projection \
            else nn.Identity()
        feature_extractor = build_feature_extractor(cfg.feature_extractor)
        encoder =  build_encoder(cfg.encoder)
        self.model = MultichannelAPC(input_dim=cfg.input_dim,
                                     feature_extractor=feature_extractor,
                                     feature_dropout=cfg.feature_dropout,
                                     feature_projection=feature_projection,
                                     encoder=encoder,
                                     encoder_embedding_dim=cfg.encoder_embedding_dim)

    def forward(self, x, lengths, target):
        prediction, target, lengths = self.model(x, lengths=lengths, target=target)
        return prediction, target, lengths


def build_model(config):
    logger.info(f"Model Config: {OmegaConf.to_object(config.model)}")
    model_config = ModelConfig(**config.model)
    model = Model(model_config)
    logger.info(f"Created model with {count_parameters(model)} parameters.")
    return model
