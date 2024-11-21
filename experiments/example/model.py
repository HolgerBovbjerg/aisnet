from dataclasses import dataclass
from logging import getLogger

from omegaconf import OmegaConf
import torch
from torch import nn


from source.utils import count_parameters
from source.nnet.feature_extraction import STFT
from source.models.utils import load_partial_checkpoints


logger = getLogger(__name__)


@dataclass
class ModelConfig:
    num_features: int = 40
    feature_type: str = "log_power"
    encoder_embedding_dim: int = 64
    sample_rate: int = 16000
    encoder_layers: int = 1
    n_classes: int = 251


class Model(nn.Module):
    """
    Main module.
    """

    def __init__(self,
                 cfg: ModelConfig = ModelConfig()):
        super().__init__()
        assert cfg.feature_type != "raw", "Model does not support raw STFT features as input."
        self.stft = STFT(n_fft=cfg.num_features*2-1, window_length=400, hop_length=400, sample_rate=cfg.sample_rate,
                         output_type=cfg.feature_type)
        input_size = cfg.num_features*2 if "power_phase" in cfg.feature_type else cfg.num_features
        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=cfg.encoder_embedding_dim,
                               num_layers=cfg.encoder_layers,
                               batch_first=True)
        self.linear = nn.Linear(cfg.encoder_embedding_dim*2, cfg.n_classes)

    def forward(self, x, lengths=None):
        x = self.stft(x).squeeze()
        x, _ = self.encoder(x)
        x = self.linear(torch.cat([torch.mean(x, dim=1), torch.std(x, dim=1)], dim=-1))
        return x

    def embed(self, x, lengths=None):
        x = self.stft(x).squeeze()
        x, _ = self.encoder(x)
        return torch.nn.functional.normalize(torch.mean(x, dim=1), p=2.0, dim=0)


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
