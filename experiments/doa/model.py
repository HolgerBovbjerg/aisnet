from dataclasses import dataclass, asdict
from logging import getLogger
from typing import Tuple, Union, List, Optional
from math import ceil

from omegaconf import OmegaConf, DictConfig
import torch
from torch import nn

from source.utils import count_parameters
from source.nnet.feature_extraction import build_feature_extractor, FeatureExtractorConfig
from source.nnet.encoder import build_encoder, EncoderConfig
from source.models.pretrained import S3PRLFeaturizer, S3PRLFeaturizerConfig
from source.models.utils import load_partial_checkpoints
from source.nnet.modules.input_projection import InputProjection


logger = getLogger(__name__)


def list_to_tensor(input_list):
    return torch.tensor(input_list, dtype=torch.float)


@dataclass
class DoaModelConfig:
    feature_extractor: Union[FeatureExtractorConfig, dict]
    encoder: Union[EncoderConfig, dict]
    microphone_array: Tuple[List[float], ...] = ([0, -0.09, 0], [0, 0.09, 0])
    elevation_resolution: float = 10.
    azimuth_resolution: float = 5.
    elevation_range: Tuple[float, float] = (90., 90.)
    azimuth_range: Tuple[float, float] = (-90., 90.)
    feature_dim: int = 40
    feature_channels: int = 2
    feature_projection: bool = True
    feature_dropout: float = 0.
    feature_dropout_first: bool = False
    encoder_embedding_dim: int = 512
    sample_rate: int = 16000
    num_channels: int = 2

    def __post_init__(self):
        if isinstance(self.encoder, (dict, DictConfig)):
            self.encoder = EncoderConfig(**self.encoder)
        elif not isinstance(self.encoder, EncoderConfig):
            raise ValueError("Wrong input for 'encoder'.")
        if isinstance(self.feature_extractor, (dict, DictConfig)):
            self.feature_extractor = FeatureExtractorConfig(**self.feature_extractor)
        elif not isinstance(self.feature_extractor, FeatureExtractorConfig):
            raise ValueError("Wrong input for 'feature_extractor'.")


class DoaModel(nn.Module):
    """
    DoA model main module.
    """

    def __init__(self,
                 cfg: DoaModelConfig):
        super().__init__()

        self.azimuth_resolution = cfg.azimuth_resolution
        self.elevation_resolution = cfg.elevation_resolution
        self.elevation_range = cfg.elevation_range
        self.azimuth_range = cfg.azimuth_range
        self.num_channels = cfg.num_channels
        self.microphone_array = list_to_tensor(cfg.microphone_array)
        self.n_mics = len(self.microphone_array)
        assert self.n_mics == self.num_channels, "Microphone array elements and channels should be the same"
        self.feature_dim = cfg.feature_dim
        self.feature_channels = cfg.feature_channels
        self.sample_rate = cfg.sample_rate
        self.feature_extractor = build_feature_extractor(cfg.feature_extractor)
        self.feature_projection = InputProjection(self.feature_dim * self.feature_channels, cfg.encoder_embedding_dim, dropout_rate=cfg.feature_dropout, dropout_first=cfg.feature_dropout_first) \
            if cfg.feature_projection else nn.Identity()
        self.encoder = build_encoder(cfg.encoder)
        self.elevation_angles, self.azimuth_angles = self._get_elevation_and_azimuth_angles()
        self.classifier = nn.Linear(cfg.encoder_embedding_dim, len(self.elevation_angles) * len(self.azimuth_angles))

    def _get_elevation_and_azimuth_angles(self):
        elevation_span = abs(self.elevation_range[1] - self.elevation_range[0])
        num_directions_elevation = ceil(elevation_span / self.elevation_resolution) if elevation_span else 1
        if elevation_span:
            elevation_angles = torch.arange(num_directions_elevation) * self.elevation_resolution + \
                               self.elevation_range[0]
        else:
            elevation_angles = torch.tensor([self.elevation_range[0]])
        azimuth_span = abs(self.azimuth_range[1] - self.azimuth_range[0])
        num_directions_azimuth = ceil(azimuth_span / self.azimuth_resolution) if azimuth_span else 1
        if azimuth_span:
            azimuth_angles = torch.arange(num_directions_azimuth) * self.azimuth_resolution + self.azimuth_range[0]
        else:
            azimuth_angles = torch.tensor([self.azimuth_range[0]])

        return elevation_angles, azimuth_angles

    def forward(self, x, lengths=None):
        init_length = x.size(-1)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        # Stack channels
        if len(x.size()) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
        # Project features to input dim of encoder
        x = self.feature_projection(x)
        # Compute new lengths after feature extraction
        extra = init_length % x.size(-2)
        if extra > 0:
            lengths = lengths - extra
        lengths = lengths // (init_length // x.size(-2))

        x, _ = self.encoder(x, lengths)
        x = self.classifier(x)
        return x, lengths


@dataclass
class DoaPretrainedModelConfig:
    feature_extractor: Union[S3PRLFeaturizerConfig, dict]
    microphone_array: Tuple[List[float], ...] = ([0, -0.09, 0], [0, 0.09, 0])
    elevation_resolution: float = 10.
    azimuth_resolution: float = 5.
    elevation_range: Tuple[float, float] = (90., 90.)
    azimuth_range: Tuple[float, float] = (-90., 90.)
    feature_dim: int = 768
    embedding_dim: int = 256
    feature_channels: int = 2
    feature_projection: bool = True
    feature_dropout: float = 0.
    feature_dropout_first: bool = False
    sample_rate: int = 16000
    num_channels: int = 2

    def __post_init__(self):
        if isinstance(self.feature_extractor, (dict, DictConfig)):
            self.feature_extractor = S3PRLFeaturizerConfig(**self.feature_extractor.config)
        elif not isinstance(self.feature_extractor, S3PRLFeaturizerConfig):
            raise ValueError("Wrong input for 'feature_extractor'.")


class DoaPretrainedModel(nn.Module):
    """
    Pretrained DoA model main module.
    """

    def __init__(self,
                 cfg: DoaPretrainedModelConfig):
        super().__init__()

        self.azimuth_resolution = cfg.azimuth_resolution
        self.elevation_resolution = cfg.elevation_resolution
        self.elevation_range = cfg.elevation_range
        self.azimuth_range = cfg.azimuth_range
        self.num_channels = cfg.num_channels
        self.microphone_array = list_to_tensor(cfg.microphone_array)
        self.n_mics = len(self.microphone_array)
        assert self.n_mics == self.num_channels, "Microphone array elements and channels should be the same"
        self.feature_dim = cfg.feature_dim
        self.feature_channels = cfg.feature_channels
        self.sample_rate = cfg.sample_rate
        self.feature_extractor = S3PRLFeaturizer(**asdict(cfg.feature_extractor))
        self.feature_projection = InputProjection(self.feature_dim * self.feature_channels, cfg.embedding_dim, dropout_rate=cfg.feature_dropout, dropout_first=cfg.feature_dropout_first) \
            if cfg.feature_projection else nn.Identity()
        self.elevation_angles, self.azimuth_angles = self._get_elevation_and_azimuth_angles()
        self.classifier = nn.Linear(cfg.embedding_dim, len(self.elevation_angles) * len(self.azimuth_angles))

    def _get_elevation_and_azimuth_angles(self):
        elevation_span = abs(self.elevation_range[1] - self.elevation_range[0])
        num_directions_elevation = ceil(elevation_span / self.elevation_resolution) if elevation_span else 1
        if elevation_span:
            elevation_angles = torch.arange(num_directions_elevation) * self.elevation_resolution + \
                               self.elevation_range[0]
        else:
            elevation_angles = torch.tensor([self.elevation_range[0]])
        azimuth_span = abs(self.azimuth_range[1] - self.azimuth_range[0])
        num_directions_azimuth = ceil(azimuth_span / self.azimuth_resolution) if azimuth_span else 1
        if azimuth_span:
            azimuth_angles = torch.arange(num_directions_azimuth) * self.azimuth_resolution + self.azimuth_range[0]
        else:
            azimuth_angles = torch.tensor([self.azimuth_range[0]])

        return elevation_angles, azimuth_angles

    def forward(self, x, lengths=None):
        # Extract pretrained model features
        features_out = []
        feature_lengths = None
        for i in range(self.num_channels):
            features, feature_lengths = self.feature_extractor(x[:, i], lengths)
            features_out.append(features)
        # Stack channels
        x = torch.cat(features_out, dim=-1)
        lengths = feature_lengths

        # Project features
        x = self.feature_projection(x)

        # Classify from pretrained model features
        x = self.classifier(x)
        return x, lengths


def build_model(config):
    logger.info(f"Model Config: {OmegaConf.to_object(config.model)}")
    if config.model.feature_extractor.name == "pretrained":
        model_config = DoaPretrainedModelConfig(**config.model)
        model = DoaPretrainedModel(model_config)
    else:
        model_config = DoaModelConfig(**config.model)
        model = DoaModel(model_config)
    logger.info(f"Created model with {count_parameters(model)} parameters.")
    # Load partial checkpoints if specified in the config
    partial_checkpoints = config.model.get("partial_checkpoints", {})
    if partial_checkpoints:
        logger.info("Loading checkpoints specified in config...")
        model = load_partial_checkpoints(model, partial_checkpoints)
        logger.info("Finished loading checkpoints.")
    return model
