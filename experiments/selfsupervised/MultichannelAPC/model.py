from dataclasses import dataclass
from logging import getLogger

from omegaconf import OmegaConf
import torch
from torch import nn


from source.utils import count_parameters
from source.nnet import STFT


logger = getLogger(__name__)


@dataclass
class ModelConfig:
    num_filters: int = 10
    encoder_embedding_dim: int = 10
    sample_rate: int = 16000
    encoder_layers: int = 1


class Model(nn.Module):
    """
    Main module.
    """

    def __init__(self,
                 cfg: ModelConfig = None):
        super().__init__()
        #self.leaf = LEAF(num_filters=cfg.num_filters, sample_rate=cfg.sample_rate)
        self.stft = STFT(n_fft=cfg.num_filters*2-1, window_length=400, hop_length=400, sample_rate=cfg.sample_rate)
        self.encoder = nn.LSTM(input_size=cfg.num_filters,
                               hidden_size=cfg.encoder_embedding_dim,
                               num_layers=cfg.encoder_layers,
                               batch_first=True)
        self.linear = nn.Linear(cfg.encoder_embedding_dim, 40)

    def forward(self, x, lengths=None):
        x = torch.log10(self.stft(x).abs() ** 2 + 1.e-9).squeeze()
        x, _ = self.encoder(x)
        x = self.linear(x[:, -1, :])
        return x


def build_model(config):
    logger.info(f"Model Config: {OmegaConf.to_object(config.model)}")
    model_config = ModelConfig(**config.model)
    model = Model(model_config)
    logger.info(f"Created model with {count_parameters(model)} parameters.")
    return model
