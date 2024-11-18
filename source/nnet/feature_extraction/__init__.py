from dataclasses import dataclass, field
from typing import Union

from .STFT import STFT, STFTConfig
from .LogMel import LogMel, LogMelConfig
from .Gabor import Gabor, ComplexGaborFilter, RealImagGaborFilter, GaborConfig
from .GCC import GCC, GCCConfig
from .SRP import SRP, SRPConfig
from .LEAF import LEAF, LEAFConfig
from .CNNFeatureExtractor import CNNFeatureExtractor, CNNFeatureExtractorConfig


feature_extractor_classes = {
    "STFT": STFT,
    "LogMel": LogMel,
    "GCC": GCC,
    "SRP": SRP,
    "LEAF": LEAF,
    "CNNFeatureExtractor": CNNFeatureExtractor,
    "Gabor": GaborConfig
}


@dataclass
class FeatureExtractorConfig:
    name: str = "cnn"
    config: Union[STFTConfig, CNNFeatureExtractorConfig, LogMelConfig, GaborConfig, GCCConfig, SRPConfig, LEAFConfig] \
        = field(default_factory=CNNFeatureExtractorConfig)


def build_extractor(config: FeatureExtractorConfig):
    try:
        return feature_extractor_classes[config.name](**config.config)
    except KeyError as e:
        raise KeyError(f"Unknown feature extractor type: {config.name}") from e
