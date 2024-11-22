from dataclasses import dataclass, asdict
from typing import Union, Dict, Type

from omegaconf import DictConfig

from .ConformerEncoder import ConformerEncoder, ConformerEncoderConfig
from .LSTMEncoder import LSTMEncoder, LSTMEncoder2, LSTMEncoderConfig
from .xLSTMEncoder import xLSTMEncoder, xLSTMEncoderConfig


encoder_classes: Dict[str, Type] = {
    "conformer": ConformerEncoder,
    "lstm": LSTMEncoder,
    "xlstm": xLSTMEncoder
}

encoder_config_classes: Dict[str, Type] = {
    "lstm": LSTMEncoderConfig,
    "conformer": ConformerEncoderConfig,
    "xlstm": xLSTMEncoderConfig,
}

@dataclass
class EncoderConfig:
    name: str  # One of: "lstm", "conformer", "xlstm"
    config: Union[LSTMEncoderConfig, ConformerEncoderConfig, xLSTMEncoderConfig]

    def __post_init__(self):
        if isinstance(self.config, (dict, DictConfig)):
            if self.name not in encoder_config_classes:
                raise ValueError(f"Unknown encoder name: {self.name}")
            self.config = encoder_config_classes[self.name](**self.config)


def build_encoder(config: EncoderConfig):
    # Validate that the name exists in the encoder_classes mapping
    if config.name not in encoder_classes:
        raise KeyError(f"Unknown encoder type: {config.name}")
    # Get the corresponding encoder class
    encoder_class = encoder_classes[config.name]
    # Use the correct config class to instantiate the encoder
    encoder_config = config.config
    return encoder_class(**asdict(encoder_config))
