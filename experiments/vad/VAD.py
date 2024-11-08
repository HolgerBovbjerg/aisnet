import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.Conformer import Conformer

class LSTMVAD(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers=2, out_dim=2):
        """PersonalVAD class initializer.
        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
            use_fc (bool, optional): Specifies, whether the model should use the
                last fully-connected hidden layer. Defaults to True.
            linear (bool, optional): Specifies the activation function used by the last
                hidden layer. If False, the tanh is used, if True, no activation is
                used. Defaults to False.
        """

        super(LSTMVAD, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        # define the model encoder...
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # use the original PersonalVAD configuration with one additional layer
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x,lengths, hidden=None, output_hidden=None):
        """VAD model forward pass method."""

        # Pass through lstm
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.encoder(x_packed, hidden)
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)

        # Project to output dimensionality
        out_padded = self.fc(out_padded)

        output = (out_padded, lengths)
        if output_hidden:
            output += (hidden,)
        return output


class ConformerVAD(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64, ffn_dim=512, num_heads=8, num_layers=2,
                 kernel_size=31, dropout=0.1, causal=True, use_group_norm=False, convolution_first=False,
                 attention_type="RelPosMHAXL", left_context: int = None, right_context: int = None,
                 out_dim: int = 2):
        super(ConformerVAD, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.encoder = Conformer(input_dim=hidden_dim, ffn_dim=ffn_dim, num_layers=num_layers,
                                 num_heads=num_heads, depthwise_conv_kernel_size=kernel_size,
                                 causal=causal, left_context=left_context, right_context=right_context,
                                 dropout=dropout, use_group_norm=use_group_norm,
                                 convolution_first=convolution_first, attention_type=attention_type)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, lengths, hidden=None, output_hidden=None):
        """VAD model forward pass method."""

        # Pass through input projection
        x = self.input_projection(x)

        # Encode in conformer
        x, _ = self.encoder(x, lengths=lengths)

        # Project to output dimensionality
        out = self.fc(x)

        output = (out, lengths)
        if output_hidden:
            output += (hidden,)
        return output
