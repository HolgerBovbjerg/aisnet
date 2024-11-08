from dataclasses import dataclass
from typing import Tuple

from torch import nn
from xlstm.blocks.slstm.layer import sLSTMLayer
from xlstm.blocks.mlstm.layer import mLSTMLayer
from xlstm.components.feedforward import create_feedforward
import xlstm.components.ln as ln


class xLSTMEncoder(nn.Module):
    def __init__(self, layers, sLSTM_config=None, mLSTM_config=None, ffn_config=None):
        super().__init__()
        self.layers = layers
        embedding_dim = (mLSTM_config.embedding_dim if mLSTM_config is not None else sLSTM_config.embedding_dim)
        self.xlstm_norm = nn.ModuleList()
        self.xlstm_blocks = nn.ModuleList()
        self.ffn_norm = nn.ModuleList()
        self.ffn = nn.ModuleList()
        if sLSTM_config is not None:
            sLSTM_config.__post_init__()
        if mLSTM_config is not None:
            mLSTM_config.__post_init__()
        if ffn_config is not None:
            ffn_config.__post_init__()
        for i, _ in enumerate(layers):
            self.xlstm_norm.append(ln.LayerNorm(ndim=embedding_dim, weight=True, bias=False))
            if layers[i] == 's':
                self.xlstm_blocks.append(sLSTMLayer(sLSTM_config))
            else:
                self.xlstm_blocks.append(mLSTMLayer(mLSTM_config))
            self.ffn_norm.append(ln.LayerNorm(ndim=embedding_dim, weight=True, bias=False))
            self.ffn.append(create_feedforward(ffn_config))
        self.post_blocks_norm = ln.LayerNorm(ndim=embedding_dim)
        self.reset_parameters()

    def forward(self, x, hidden):
        if hidden is None:
            hidden = {}
        for block_idx, block in enumerate(self.xlstm_blocks):
            if self.layers[block_idx] == 's':
                x, hidden[f'block_{block_idx}'] = block(self.xlstm_norm[block_idx](x),
                                                        hidden.get(f'block_{block_idx}', None), return_last_state=True)
            else:
                x = block(self.xlstm_norm[block_idx](x))
            x = x + self.ffn[block_idx](self.ffn_norm[block_idx](x))
        x = self.post_blocks_norm(x)
        return x, hidden

    def reset_parameters(self):
        for i in range(len(self.layers)):
            self.xlstm_norm[i].reset_parameters()
            self.xlstm_blocks[i].reset_parameters()
            self.ffn_norm[i].reset_parameters()
            self.ffn[i].reset_parameters()
        self.post_blocks_norm.reset_parameters()
