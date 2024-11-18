from dataclasses import dataclass, asdict
from typing import Tuple

from torch import nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


@dataclass
class mLSTMConfig:
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4

@dataclass
class sLSTMConfig:
    backend: str = "cuda"
    num_heads: int = 4
    conv1d_kernel_size: int = 4
    bias_init: str = "powerlaw_blockdependent"

@dataclass
class sLSTMFeedForwardConfig:
    proj_factor: float = 1.3
    act_fn: str = "gelu"

@dataclass
class xLSTMEncoderConfig:
    input_dim: int = 40
    embedding_dim: int = 256
    slstm_config = sLSTMConfig
    mlstm_config = mLSTMConfig
    slstm_feedforward_config = sLSTMFeedForwardConfig
    context_length: int = 256
    num_blocks: int = 7
    slstm_at: Tuple[int] = (1,)

    def __post_init__(self):
        self.slstm_config = sLSTMConfig(**self.slstm_config)
        self.mlstm_config = mLSTMConfig(**self.mlstm_config)
        self.slstm_feedforward_config = sLSTMFeedForwardConfig


class xLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int = 40, embedding_dim: int = 256, slstm_config = sLSTMConfig(),
                 mlstm_config = mLSTMConfig(), slstm_feedforward_config=sLSTMFeedForwardConfig(),
                 context_length: int = 256, num_blocks: int = 7, slstm_at: Tuple[int] = (1,)):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    **asdict(mlstm_config)
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    **asdict(slstm_config)
                ),
                feedforward=FeedForwardConfig(**asdict(slstm_feedforward_config)),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=list(slstm_at),
        )
        self.encoder = xLSTMBlockStack(cfg)

    def forward(self, x, padding_mask, hidden=None, output_hidden_states: bool = False):
        x = self.input_projection(x)
        x = self.encoder(x)
        return x

    def reset_parameters(self):
        for i in range(len(self.layers)):
            self.xlstm_norm[i].reset_parameters()
            self.xlstm_blocks[i].reset_parameters()
            self.ffn_norm[i].reset_parameters()
            self.ffn[i].reset_parameters()
        self.post_blocks_norm.reset_parameters()
