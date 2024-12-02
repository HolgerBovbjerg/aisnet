from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _padding_mask_to_lengths(padding_mask: torch.Tensor) -> torch.Tensor:
    return padding_mask.size(1) - padding_mask.sum(dim=1)


@dataclass
class LSTMEncoderConfig:
    input_dim: int = 40
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.
    projection_size: Optional[int] = 0
    bias: bool = True
    bidirectional: bool = False
    batch_first: bool = True


class LSTMEncoder(nn.Module):
    """
    LSTMEncoder class.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0., projection_size=0, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout,
                            proj_size=projection_size, **kwargs)

    def forward(self, x, lengths, hidden=None, output_hidden: bool = False):
        # first pack the padded sequences
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # lstm pass
        out_packed, hidden = self.lstm(x_packed, hidden)
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)
        output = (out_padded, lengths)
        if output_hidden:
            output += (hidden,)

        return output


class LSTMEncoder2(nn.Module):
    """
    LSTMEncoder2 adds support for access to hidden states after each layer. Slower than LSTMEncoder.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0, projection_size=0, **kwargs):
        super().__init__()
        in_sizes = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_sizes = [hidden_dim] * num_layers
        self.lstms = nn.ModuleList(
            [nn.LSTM(input_size=in_size, hidden_size=out_size, batch_first=True, proj_size=projection_size, **kwargs)
             for in_size, out_size in zip(in_sizes, out_sizes)]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None


    def forward(self, x, lengths, hidden=None, output_hidden_states: bool = False):
        # Pack padded sequences for efficient computation
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed = x_packed
        hidden_states = []

        for i, lstm in enumerate(self.lstms):
            # Pass through the current LSTM layer
            out_packed, _ = lstm(out_packed, hidden)
            # Unpack the sequence for potential dropout or further processing
            out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)

            # Apply dropout except on the last layer
            if self.dropout and i + 1 < len(self.lstms):
                out_padded = self.dropout(out_padded)

            # Store intermediate hidden states if required
            if output_hidden_states:
                hidden_states.append(out_padded)

            if i + 1 < len(self.lstms):
                # Pack again for the next LSTM layer
                out_packed = pack_padded_sequence(out_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Return the output and optionally the hidden states
        output = (out_padded, lengths)
        if output_hidden_states:
            output += (hidden_states,)
        return output
