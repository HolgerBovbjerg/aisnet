from dataclasses import dataclass, field
from typing import Union

from .stft import STFT, STFTConfig
from .logmel import LogMel, LogMelConfig
from .gabor import Gabor, ComplexGaborFilter, RealImagGaborFilter, GaborConfig
from .gcc import GCC, GCCConfig
from .srp import SRP, SRPConfig
from .leaf import LEAF, LEAFConfig
from .cnn_feature_extractor import CNNFeatureExtractor, CNNFeatureExtractorConfig


feature_extractor_classes = {
    "stft": STFT,
    "logmel": LogMel,
    "gcc": GCC,
    "srp": SRP,
    "leaf": LEAF,
    "cnn_feature_extractor": CNNFeatureExtractor,
    "gabor": GaborConfig
}


@dataclass
class FeatureExtractorConfig:
    name: str = "cnn"
    config: Union[STFTConfig, CNNFeatureExtractorConfig, LogMelConfig, GaborConfig, GCCConfig, SRPConfig, LEAFConfig] \
        = field(default_factory=CNNFeatureExtractorConfig)


def build_feature_extractor(config: FeatureExtractorConfig):
    try:
        return feature_extractor_classes[config.name](**config.config)
    except KeyError as e:
        raise KeyError(f"Unknown feature extractor type: {config.name}") from e
