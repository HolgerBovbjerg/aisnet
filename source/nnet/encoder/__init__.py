from dataclasses import dataclass, field
from typing import Union

from .ConformerEncoder import ConformerEncoder, ConformerEncoderConfig
from .LSTMEncoder import LSTMEncoder, LSTMEncoder2, LSTMEncoderConfig
from .xLSTMEncoder import xLSTMEncoder, xLSTMEncoderConfig


encoder_classes = {
    "conformer": ConformerEncoder,
    "lstm": LSTMEncoder,
    "xlstm": xLSTMEncoder
}


@dataclass
class EncoderConfig:
    name: str = "Conformer"
    config: Union[LSTMEncoderConfig, ConformerEncoderConfig, xLSTMEncoderConfig] \
        = field(default_factory=ConformerEncoderConfig)


def build_encoder(config: EncoderConfig):
    try:
        return encoder_classes[config.name](**config.config)
    except KeyError as e:
        raise KeyError(f"Unknown encoder type: {config.name}") from e
