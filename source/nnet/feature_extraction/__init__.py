from dataclasses import dataclass, asdict
from typing import Union

from omegaconf import DictConfig

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
    "gabor": Gabor
}

feature_extractor_config_classes = {
    "stft": STFTConfig,
    "logmel": LogMelConfig,
    "gcc": GCCConfig,
    "srp": SRPConfig,
    "leaf": LEAFConfig,
    "cnn_feature_extractor": CNNFeatureExtractorConfig,
    "gabor": GaborConfig
}


@dataclass
class FeatureExtractorConfig:
    name: str
    config: Union[STFTConfig, CNNFeatureExtractorConfig, LogMelConfig, GaborConfig, GCCConfig, SRPConfig, LEAFConfig]

    def __post_init__(self):
        # If `config` is a dictionary, convert it into the appropriate class
        if isinstance(self.config, (dict, DictConfig)):
            if self.name not in feature_extractor_config_classes:
                raise ValueError(f"Unknown encoder name: {self.name}")
            self.config = feature_extractor_config_classes[self.name](**self.config)


def build_feature_extractor(config: FeatureExtractorConfig):
    # Validate that the name exists in the feature_extractor_classes mapping
    if config.name not in feature_extractor_classes:
        raise KeyError(f"Unknown feature_extractor type: {config.name}")
    # Get the corresponding feature_extractor class
    feature_extractor_class = feature_extractor_classes[config.name]
    # Use the correct config class to instantiate the feature_extractor
    feature_extractor_config = config.config
    return feature_extractor_class(**asdict(feature_extractor_config))
