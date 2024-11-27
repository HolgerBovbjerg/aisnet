from dataclasses import dataclass, field
from logging import getLogger
from typing import Tuple, Union, List, Optional

from omegaconf import OmegaConf, DictConfig
import torch
from torch import nn

from source.utils import count_parameters
from source.nnet.feature_extraction import FeatureExtractorConfig, build_feature_extractor, GCCConfig
from source.nnet.encoder import EncoderConfig, build_encoder
from source.nnet.utils.masking import MaskGeneratorConfig, MaskGenerator
from source.models.utils import load_partial_checkpoints
from source.models.selfsupervised.sslsar import SSLSAR

logger = getLogger(__name__)

@dataclass
class ModelConfig:
    feature_dim: int
    spectral_encoder_embedding_dim: int
    spatial_encoder_embedding_dim: int
    n_feature_channels: int = 2
    feature_projection: bool = True
    feature_dropout: float = 0.
    feature_dropout_first: bool = False
    feature_extractor: Optional[Union[FeatureExtractorConfig, dict]] = None
    mask_generator: Optional[Union[MaskGeneratorConfig, dict]] = None
    spatial_encoder: Optional[Union[EncoderConfig, dict]] = None
    spectral_encoder: Optional[Union[EncoderConfig, dict]] = None
    decoder: Optional[nn.Module] = None
    sample_rate: int = 16000

    def __post_init__(self):
        if isinstance(self.spatial_encoder, (dict, DictConfig)):
            self.spatial_encoder = EncoderConfig(**self.spatial_encoder)
        elif not isinstance(self.spatial_encoder, EncoderConfig):
            raise ValueError("Wrong input for 'encoder'.")
        if isinstance(self.spectral_encoder, (dict, DictConfig)):
            self.spectral_encoder = EncoderConfig(**self.spectral_encoder)
        elif not isinstance(self.spectral_encoder, EncoderConfig):
            raise ValueError("Wrong input for 'encoder'.")
        if isinstance(self.feature_extractor, (dict, DictConfig)):
            self.feature_extractor = FeatureExtractorConfig(**self.feature_extractor)
        elif not isinstance(self.feature_extractor, FeatureExtractorConfig):
            raise ValueError("Wrong input for 'feature_extractor'.")
        if isinstance(self.mask_generator, (dict, DictConfig)):
            self.gcc_config = MaskGeneratorConfig(**self.mask_generator)
        elif not isinstance(self.gcc_config, MaskGeneratorConfig):
            raise ValueError("Wrong input for 'mask_generator'.")


class Model(nn.Module):
    """
    Main module.
    """

    def __init__(self,
                 cfg: ModelConfig):
        super().__init__()
        feature_extractor = build_feature_extractor(cfg.feature_extractor)
        mask_generator = MaskGenerator(cfg.mask_generator)
        spatial_encoder = build_encoder(cfg.spatial_encoder)
        spectral_encoder = build_encoder(cfg.spectral_encoder)
        self.model = SSLSAR(feature_extractor=feature_extractor,
                            feature_dim=cfg.feature_dim,
                            mask_generator=mask_generator,
                            spatial_encoder=spatial_encoder,
                            spectral_encoder=spectral_encoder,
                            spectral_encoder_embedding_dim=cfg.spectral_encoder_embedding_dim,
                            spatial_encoder_embedding_dim=cfg.spatial_encoder_embedding_dim,
                            feature_projection=cfg.feature_projection,
                            feature_dropout_first=cfg.feature_dropout_first,
                            feature_dropout=cfg.feature_dropout,
                            n_feature_channels=cfg.n_feature_channels)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor,
                target: Optional[torch.Tensor] = None):
        return self.model(x, lengths, target)


def build_model(config):
    logger.info(f"Model Config: {OmegaConf.to_object(config.model)}")
    model_config = ModelConfig(**config.model)
    model = Model(model_config)
    logger.info(f"Created model with {count_parameters(model)} parameters.")
    # Load partial checkpoints if specified in the config
    partial_checkpoints = config.model.get("partial_checkpoints", {})
    if partial_checkpoints:
        logger.info("Loading checkpoints specified in config...")
        model = load_partial_checkpoints(model, partial_checkpoints)
        logger.info("Finished loading checkpoints.")
    return model
