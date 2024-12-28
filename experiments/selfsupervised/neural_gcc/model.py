from dataclasses import dataclass, field
from logging import getLogger
from typing import Tuple, Union, List, Optional

from omegaconf import OmegaConf, DictConfig
import torch
from torch import nn


from source.utils import count_parameters
from source.nnet.feature_extraction import FeatureExtractorConfig, build_feature_extractor, GCCConfig
from source.nnet.encoder import EncoderConfig, build_encoder
from source.models.utils import load_partial_checkpoints
from source.models.selfsupervised.neural_gcc import NeuralGCC


logger = getLogger(__name__)


@dataclass
class ModelConfig:
    feature_dim: int
    gcc_dim: int
    encoder_embedding_dim: int
    encoder_num_layers: int
    n_feature_channels: int = 1
    feature_projection: bool = True
    feature_dropout: float = 0.
    feature_dropout_first: bool = False
    feature_extractor: Optional[Union[FeatureExtractorConfig, dict]] = None
    encoder: Optional[Union[EncoderConfig, dict]] = None
    gcc_config: Optional[Union[GCCConfig, dict]] = None
    sample_rate: int = 16000
    normalize_target: bool = True

    def __post_init__(self):
        if isinstance(self.encoder, (dict, DictConfig)):
            self.encoder = EncoderConfig(**self.encoder)
        elif not isinstance(self.encoder, EncoderConfig):
            raise ValueError("Wrong input for 'encoder'.")
        if isinstance(self.feature_extractor, (dict, DictConfig)):
            self.feature_extractor = FeatureExtractorConfig(**self.feature_extractor)
        elif not isinstance(self.feature_extractor, FeatureExtractorConfig):
            raise ValueError("Wrong input for 'feature_extractor'.")
        if isinstance(self.gcc_config, (dict, DictConfig)):
            self.gcc_config = GCCConfig(**self.gcc_config)
        elif not isinstance(self.gcc_config, GCCConfig):
            raise ValueError("Wrong input for 'gcc_config'.")


class Model(nn.Module):
    """
    Main module.
    """

    def __init__(self,
                 cfg: ModelConfig):
        super().__init__()
        feature_extractor = build_feature_extractor(cfg.feature_extractor)
        encoder = build_encoder(cfg.encoder)
        self.model = NeuralGCC(feature_extractor=feature_extractor,
                               feature_dim=cfg.feature_dim,
                               encoder=encoder,
                               encoder_embedding_dim=cfg.encoder_embedding_dim,
                               gcc_config=cfg.gcc_config,
                               gcc_dim=cfg.gcc_dim,
                               feature_projection=cfg.feature_projection,
                               feature_dropout_first=cfg.feature_dropout_first,
                               feature_dropout=cfg.feature_dropout,
                               n_feature_channels=cfg.n_feature_channels,
                               normalize_target=cfg.normalize_target,)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                time_shift: int = 0):
        return self.model(x, lengths, target, time_shift)

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
