import math
from typing import Tuple, Optional
from dataclasses import dataclass, field, asdict

import torch
from torch import nn
import torch.nn.functional as F
from speechbrain.nnet.attention import MultiheadAttention

from source.nnet.utils.masking import lengths_to_padding_mask, generate_attention_mask


class RelPosMHAXL(nn.Module):
    """ This class implements the relative multihead implementation similar to that in Transformer XL
    https://arxiv.org/pdf/1901.02860.pdf

    Arguments
    ---------
    embed_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    num_heads: int
        Number of attention heads.
    dropout : float, optional
        Dropout rate.
    vbias: bool, optional
        Whether to use bias for computing value.
    vdim: int, optional
        Size for value. Default is embed_dim (Note each head is embed_dim // num_heads).
    mask_pos_future: bool, optional
        Whether to mask future positional encodings values.
        Must be true for causal applications e.g. decoder.
    Example
    -------
    >>> inputs = torch.rand([6, 60, 512])
    >>> pos_emb = torch.rand([1, 2*60-1, 512])
    >>> net = RelPosMHAXL(num_heads=8, embed_dim=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs, pos_emb)
    >>> outputs.shape
    torch.Size([6, 60, 512])
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        vbias=False,
        vdim=None,
        mask_pos_future=False,
    ):
        super(RelPosMHAXL, self).__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.vdim == embed_dim
        self.mask_pos_future = mask_pos_future
        self.vbias = vbias

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.vhead_dim = self.vdim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert (
            self.vhead_dim * num_heads == self.vdim
        ), "vdim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.qk_proj_weight = nn.Parameter(
                torch.empty(2 * embed_dim, embed_dim)
            )
            self.v_proj_weight = nn.Parameter(torch.empty(self.vdim, embed_dim))
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty(3 * embed_dim, embed_dim)
            )

        if vbias:
            self.value_bias_weight = nn.Parameter(torch.empty(self.vdim))
        else:
            self.vbias = None

        self.dropout_att = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.vdim, embed_dim)

        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_bias_u = nn.Parameter(
            torch.empty(self.head_dim, self.num_heads)
        )
        self.pos_bias_v = nn.Parameter(
            torch.empty(self.head_dim, self.num_heads)
        )

        if next(self.parameters()).dtype == torch.float16:
            self.attn_fill_value = -65000
        else:
            dtype = next(self.parameters()).dtype
            self.attn_fill_value = torch.finfo(dtype).min

        self._reset_parameters()
        self.scale = 1 / math.sqrt(self.embed_dim)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.qk_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.vbias is not None:
            torch.nn.init.constant_(self.value_bias_weight, 0.0)

        # positional biases
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Relative shift implementation."""
        # batch, head, time1, 2*time1-1.

        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)

        if self.mask_pos_future:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x[..., : pos_len // 2 + 1]

    def forward(
        self,
        query,
        key,
        value,
        pos_embs,
        key_padding_mask=None,
        attn_mask=None,
        return_attn_weights=True,
    ):
        """
        Arguments
        ----------
        query : tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        pos_emb : tensor
            bidirectional sinusoidal positional embedding tensor (1, 2*S-1, E) where S is the max length between source and target sequence lengths,
            and E is the embedding dimension.
        key_padding_mask : tensor
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : tensor
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.

        Outputs
        -------
        out : tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_score : tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """

        # query, key and value are of shape batch, time, embed_dim
        bsz = query.shape[0]
        klen = key.shape[1]
        qlen = query.shape[1]

        if self._qkv_same_embed_dim:
            # self-attention
            if (query is key or torch.equal(query, key)) and (
                key is value or torch.equal(key, value)
            ):
                query, key, value = (
                    nn.functional.linear(query, self.in_proj_weight)
                    .view(bsz, -1, self.num_heads, self.head_dim * 3)
                    .chunk(3, dim=-1)
                )
            else:
                qweight, kweight, vweight = self.in_proj_weight.chunk(3, dim=0)
                query = nn.functional.linear(query, qweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                key = nn.functional.linear(key, kweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                value = nn.functional.linear(value, vweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
        else:
            raise NotImplementedError
            query, key = (
                nn.functional.linear(query, self.qk_proj_weight)
                .view(bsz, -1, self.num_heads, self.head_dim * 2)
                .chunk(2, dim=-1)
            )
            value = nn.functional.linear(value, self.v_proj_weight).view(
                bsz, -1, self.num_heads, self.vhead_dim
            )

        if self.vbias is not None:
            value = value + self.value_bias_weight.view(
                1, 1, self.num_heads, self.vhead_dim
            )

        p_k = self.linear_pos(pos_embs).view(
            1, -1, self.num_heads, self.head_dim
        )
        # (batch, head, klen, d_k)

        q_with_bias_u = (
            query + self.pos_bias_u.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        # (batch, head, qlen, d_k)
        q_with_bias_v = (
            query + self.pos_bias_v.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)

        # (batch, head, qlen, klen)
        matrix_ac = torch.matmul(q_with_bias_u, key.permute(0, 2, 3, 1))
        # (batch, num_heads, klen, 2*klen-1)
        matrix_bd = torch.matmul(q_with_bias_v, p_k.permute(0, 2, 3, 1))
        matrix_bd = self.rel_shift(matrix_bd)  # shifting trick

        # if klen != qlen:
        #   import ipdb
        #  ipdb.set_trace(

        attn_score = (matrix_ac + matrix_bd) * self.scale

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.view(1, 1, qlen, klen)
            else:
                attn_mask = attn_mask.view(-1, self.num_heads, qlen, klen)

            if attn_mask.dtype == torch.bool:
                attn_score = attn_score.masked_fill(
                    attn_mask, self.attn_fill_value
                )
            else:
                attn_score += attn_mask

        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(
                key_padding_mask.view(bsz, 1, 1, klen), self.attn_fill_value,
            )

        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = self.dropout_att(attn_score)
        x = torch.matmul(
            attn_score, value.transpose(1, 2)
        )  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.vhead_dim * self.num_heads)
        )  # (batch, time1, d_model)

        out = self.out_proj(x)
        if return_attn_weights:
            return out, attn_score
        return out


class RelPosEncXL(nn.Module):
    """

    """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

        inv_freq = torch.exp(
            torch.arange(0, self.emb_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.emb_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
        input tensor with shape batch_size, seq_len, embed_dim
        Returns
        -------
        pos_emb : torch.Tensor
        """
        seq_len = x.size(1)
        with torch.no_grad():
            tot_pe = torch.zeros((2, seq_len, self.emb_dim), dtype=x.dtype).to(
                x
            )
            pe_past = tot_pe[0]
            pe_future = tot_pe[1]
            positions = (
                torch.arange(0, seq_len, dtype=x.dtype, device=x.device)
                .to(x)
                .unsqueeze(-1)
            )
            sinusoids = torch.sin(positions * self.inv_freq)
            pe_past[:, 0::2] = sinusoids
            pe_past[:, 1::2] = torch.cos(positions * self.inv_freq)
            pe_future[:, 0::2] = sinusoids  # same for past and future
            pe_future[:, 1::2] = torch.cos(-positions * self.inv_freq)

            pe_past = torch.flip(pe_past, (0,)).unsqueeze(0)
            pe_future = pe_future[1:].unsqueeze(0)
            pe = torch.cat([pe_past, pe_future], dim=1)
            # pe is now 1, 2*seq_len, embed_dim
            return pe


class Conv2dSubSampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
            self,
            input_dim: int,
            num_channels: int,
            depthwise_kernel_size: int,
            dropout: float = 0.0,
            bias: bool = False,
            use_group_norm: bool = False,
            causal_convolution: bool = False
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.causal_convolution = causal_convolution
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.pointwise_conv1 = torch.nn.Conv1d(input_dim, 2 * num_channels, 1, stride=1, padding=0, bias=bias)
        self.glu = torch.nn.GLU(dim=1)
        self.padding = (depthwise_kernel_size - 1) if causal_convolution else (depthwise_kernel_size - 1) // 2
        self.depthwise_conv = torch.nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=depthwise_kernel_size,
            stride=1,
            padding=self.padding,
            groups=num_channels,
            bias=bias)
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels) if use_group_norm \
            else torch.nn.BatchNorm1d(num_channels)
        self.silu = torch.nn.SiLU()
        self.pointwise_conv2 = torch.nn.Conv1d(num_channels, input_dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        if self.causal_convolution:
            x = x[..., :(-1 * self.padding)]
        x = self.norm(x)
        x = self.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        if mask is not None:
            x.masked_fill_(mask, 0.0)
        return x


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(input_dim)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.silu = torch.nn.SiLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_dim, input_dim, bias=True)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        x = self.layernorm(x)
        x = self.linear1(x)
        x = self.silu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
            self,
            input_dim: int,
            ffn_dim: int,
            num_attention_heads: int,
            depthwise_conv_kernel_size: int,
            dropout: float = 0.0,
            use_group_norm: bool = False,
            convolution_first: bool = False,
            causal: bool = False,
            left_context: Optional[int] = None,
            right_context: Optional[int] = None,
            attention_type: str = "RelPosMHAXL"
    ) -> None:
        super().__init__()
        # Validate inputs
        if left_context is not None and left_context < 0:
            raise ValueError("left_context must be 0 or positive if not None.")
        if right_context is not None and right_context < 0:
            raise ValueError("right_context must be 0 or positive if not None.")
        if causal and (right_context is not None and right_context > 0):
            raise ValueError("If causal is True, right_context must be 0 or None.")
        self.causal = causal
        self.left_context = left_context
        # Set right_context
        if causal:
            self.right_context = 0  # No right context if causal
        else:
            self.right_context = right_context
        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.attention_type = attention_type
        if attention_type == "RegularMHA":
            self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_attention_heads, dropout=dropout)
        elif attention_type == "RelPosMHAXL":
            self.self_attn = RelPosMHAXL(embed_dim=input_dim, num_heads=num_attention_heads, dropout=dropout,
                                         mask_pos_future=causal)
        else:
            raise ValueError(f"Attention type - {attention_type} - not implemented")
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
            causal_convolution=causal
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = x.transpose(0, 1)
        x = self.conv_module(x, mask)
        x = x.transpose(0, 1)
        x = residual + x
        return x

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor],
                pos_embs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            x (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual

        conv_mask = None
        if key_padding_mask is not None:
            conv_mask = key_padding_mask.unsqueeze(-1).transpose(0, 1)

        if self.convolution_first:
            x = self._apply_convolution(x, conv_mask)

        residual = x
        x = self.self_attn_layer_norm(x)

        attn_mask = None
        if self.left_context or self.right_context:
            attn_mask = generate_attention_mask(size=(x.size(1), x.size(1)),
                                                 left_context=self.left_context,
                                                 right_context=self.right_context,
                                                 device=x.device)
        x, self_attention = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            pos_embs=pos_embs,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x, conv_mask)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, self_attention


@dataclass
class ConformerEncoderConfig:
    input_dim: int = 512
    ffn_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    depthwise_conv_kernel_size: int = 31
    dropout: float = 0.
    use_group_norm: bool = False
    convolution_first: bool = False
    left_context: Optional[int] = None
    right_context: Optional[int] = None
    causal: bool = False


class ConformerEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 num_layers: int,
                 depthwise_conv_kernel_size: int = 31,
                 dropout: float = 0.,
                 use_group_norm: bool = False,
                 convolution_first: bool = False,
                 left_context: Optional[int] = None,
                 right_context: Optional[int] = None,
                 causal: bool = False):
        super().__init__()
        self.positional_encoder = RelPosEncXL(emb_dim=input_dim)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                    causal=causal,
                    left_context=left_context,
                    right_context=right_context,
                    attention_type="RelPosMHAXL"
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, lengths, output_hidden_states=False, output_attentions=False):
        padding_mask = lengths_to_padding_mask(lengths)
        pos_embs = self.positional_encoder(x)
        hidden_states = []
        attentions = []
        for layer in self.conformer_layers:
            x, attention = layer(x, padding_mask, pos_embs)
            # Store intermediate results if specified
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attention)
        output = (x, lengths)
        if output_hidden_states:
            output += (hidden_states,)
        if output_attentions:
            output += (attentions,)
        return output[0] if len(output) == 1 else output
