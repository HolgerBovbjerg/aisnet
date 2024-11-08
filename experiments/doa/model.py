from dataclasses import dataclass, field, asdict
from logging import getLogger
from typing import Tuple, Union, List, Optional
from math import ceil

from omegaconf import OmegaConf
import torch
from torch import nn

from source.utils import count_parameters
from source.nnet import ConformerEncoder, LSTMEncoder, xLSTMEncoder
from source.nnet import SRPPHAT, GCCPHAT, STFT, CNNFeatureExtractor
from source.nnet import lengths_to_padding_mask

logger = getLogger(__name__)


@dataclass
class STFTConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    window_length: int = 400
    hop_length: int = 160
    window_type: str = "hann"
    pad_mode: str = "constant"


@dataclass
class SRPPHATConfig:
    microphone_positions: torch.Tensor
    n_fft: int = 512
    window_length: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    center: bool = False
    window_type: str = "hann"
    elevation_resolution: float = 5.
    azimuth_resolution: float = 5.
    elevation_range: Tuple[float, float] = (0., 180.)
    azimuth_range: Tuple[float, float] = (-180., 180.)
    c_sound: float = 343.
    frequency_range: Optional[Tuple[float, float]] = None


@dataclass
class GCCPHATConfig:
    n_fft: int
    window_length: int
    hop_length: int
    sample_rate: int
    center: bool = False
    window_type: str = "hann"
    max_delay: Optional[torch.Tensor] = None
    n_mics: int = 2


@dataclass
class CNNFeatureExtractorConfig:
    in_channels: Tuple[int, ...] = (1, 512, 512, 512, 512, 512, 512)
    out_channels: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    strides: Tuple[int, ...] = (5, 3, 2, 2, 2, 2, 2)
    stacked_consecutive_features: int = 1
    stacked_features_stride: int = 1
    causal: bool = False
    sample_rate: int = 16000


@dataclass
class FeatureExtractorConfig:
    type: str = "stft"
    config: Union[STFTConfig, SRPPHATConfig, GCCPHATConfig, CNNFeatureExtractorConfig] = (
        field(default_factory=STFTConfig))


def initialize_feature_extractor(config: FeatureExtractorConfig):
    if config.type.lower() == "stft":
        return STFT(**asdict(config.config))
    if config.type.lower() == "gccphat":
        return GCCPHAT(**asdict(config.config))
    elif config.type.lower() == "srpphat":
        return SRPPHAT(**asdict(config.config))
    elif config.type.lower() == "cnn":
        return CNNFeatureExtractor(**asdict(config.config))
    else:
        raise ValueError(f"Unknown feature extractor type: {config.type}")


@dataclass
class LSTMEncoderConfig:
    input_dim: int = 40
    hidden_dim: int = 64
    num_layers: int = 3
    bias: bool = True
    dropout: float = 0.
    projection_size: int = 0
    bidirectional: bool = False


@dataclass
class ConformerEncoderConfig:
    input_dim: int = 40
    ffn_dim: int = 512
    num_heads: int = 4
    num_layers: int = 8
    depthwise_conv_kernel_size: int = 31
    dropout: float = 0.
    use_group_norm: bool = False
    convolution_first: bool = False
    left_context: Optional[int] = 0
    right_context: Optional[int] = 0
    causal: bool = True


@dataclass
class xLSTMEncoderConfig:
    layers: 1
    sLSTM_config = None
    mLSTM_config = None
    ffn_config = None


@dataclass
class EncoderConfig:
    type: str = "conformer"
    config: Union[LSTMEncoderConfig, ConformerEncoderConfig, xLSTMEncoderConfig] \
        = field(default_factory=ConformerEncoderConfig)


def initialize_encoder(config: EncoderConfig):
    if config.type.lower() == "lstm":
        return LSTMEncoder(**asdict(config.config))
    elif config.type.lower() == "conformer":
        return ConformerEncoder(**asdict(config.config))
    elif config.type.lower() == "xlstm":
        return xLSTMEncoder(**asdict(config.config))
    else:
        raise ValueError(f"Unknown encoder type: {config.type}")


def list_to_tensor(input_list):
    return torch.tensor(input_list, dtype=torch.float)


@dataclass
class ModelConfig:
    feature_dim: int = 40
    embedding_dim: int = 512
    microphone_array: Tuple[List[float]] = ([0, -0.09, 0], [0, 0.09, 0])
    elevation_resolution: float = 10.
    azimuth_resolution: float = 5.
    elevation_range: Tuple[float, float] = (90., 90.)
    azimuth_range: Tuple[float, float] = (-90., 90.)
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def update(self, cfg: dict):
        for key, value in cfg.items():
            if key == "encoder" and isinstance(value, dict):
                config_classes = {
                    "lstm": LSTMEncoderConfig,
                    "conformer": ConformerEncoderConfig,
                    "xlstm": xLSTMEncoderConfig
                }
                self.encoder.type = value['type']
                self.encoder.config = config_classes[self.encoder.type](**value["config"])
            elif key == "feature_extractor" and isinstance(value, dict):
                config_classes = {
                    "stft": STFTConfig,
                    "gccphat": GCCPHATConfig,
                    "srpphat": SRPPHATConfig,
                    "cnn": CNNFeatureExtractorConfig,
                }
                self.feature_extractor.type = value['type']
                self.feature_extractor.config = config_classes[self.feature_extractor.type](**value["config"])

            else:
                if hasattr(self, key):
                    setattr(self, key, value)


class Model(nn.Module):
    """
    Bootstrap Predictive Coding main module.
    """

    def __init__(self,
                 cfg: ModelConfig = None):
        super().__init__()

        self.azimuth_resolution = cfg.azimuth_resolution
        self.elevation_resolution = cfg.elevation_resolution
        self.elevation_range = cfg.elevation_range
        self.azimuth_range = cfg.azimuth_range
        self.microphone_array = list_to_tensor(cfg.microphone_array)
        self.feature_dim = cfg.feature_dim
        self.n_mics = len(self.microphone_array)
        self.feature_dim = cfg.feature_dim
        self.embedding_dim = cfg.embedding_dim
        cfg.encoder.config.input_dim = self.feature_dim * self.n_mics * 2
        self.feature_extractor = initialize_feature_extractor(cfg.feature_extractor)
        self.encoder = initialize_encoder(cfg.encoder)
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

    @staticmethod
    def forward_padding_mask(
            features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes padding mask for features after feature extraction given original padding mask.
        Assumes input of size (batch_size, channels, seq_len, input_dim).
        :param features: torch.Tensor of size (batch_size, channels, orig_seq_len, input_dim)
        :param padding_mask: torch.Tensor of size (batch_size, orig_seq_len)
        :return: padding_mask: torch.Tensor of size (batch_size, seq_len)
        """
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(self, x, lengths=None):
        encoder_padding_mask = lengths_to_padding_mask(lengths)
        x = self.feature_extractor(x)
        x = torch.cat([torch.log10(x.abs() ** 2 + 1.e-9), x.angle()], dim=-1)
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        encoder_padding_mask = self.forward_padding_mask(x, encoder_padding_mask)
        x, _ = self.encoder(x, encoder_padding_mask)
        x = self.classifier(x)
        return x


def build_model(config):
    model_config = ModelConfig()
    model_config.update(OmegaConf.to_object(config.model))
    logger.info(f"Model Config: {model_config}")
    model = Model(model_config)
    logger.info(f"Created model with {count_parameters(model)} parameters.")
    return model
