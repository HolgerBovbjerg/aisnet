from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.attention import RelPosEncXL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from xlstm.blocks.slstm.layer import sLSTMLayer
from xlstm.blocks.mlstm.layer import mLSTMLayer
from xlstm.components.feedforward import create_feedforward
import xlstm.components.ln as ln

from source.nnet.modules import EMA
from source.nnet.feature_extraction import CNNFeatureExtractor, LogMel
from source.nnet.encoder.ConformerEncoder import ConformerLayer


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


def _padding_mask_to_lengths(padding_mask: torch.Tensor) -> torch.Tensor:
    return padding_mask.size(1) - padding_mask.sum(dim=1)


def compute_mask_indices(
        shape: Tuple[int, int],
        padding_mask: Optional[torch.Tensor],
        mask_probability: float,
        mask_length: int,
        mask_type: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 0,
        no_overlap: bool = False,
        min_space: int = 0,
        max_mask_percentage_per_window: float = 0.0,
        max_mask_percentage_window_size: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_probability: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from poisson distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        max_mask_percentage_per_window: maximum percentage of masked tokens within any window of size N
        window_size: size of the window to check for max_mask_percentage_per_window
    """
    batch_size, total_size = shape
    mask = np.full((batch_size, total_size), False)

    total_number_of_masks = int(
        mask_probability * total_size / float(mask_length) + np.random.rand()
    )
    total_number_of_masks = max(min_masks, total_number_of_masks)

    for i in range(batch_size):
        if padding_mask is not None:
            sequence_length = total_size - padding_mask[i].long().sum().item()
            number_of_masks = int(
                mask_probability * sequence_length / float(mask_length) + np.random.rand()
            )
            number_of_masks = max(min_masks, number_of_masks)
        else:
            sequence_length = total_size
            number_of_masks = total_number_of_masks

        if mask_type == "static":
            lengths = np.full(number_of_masks, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=number_of_masks)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=number_of_masks)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=number_of_masks)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise TypeError("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sequence_length - 1)

        mask_indices = []
        if no_overlap:
            def arrange(start, end, length, keep_length):
                span_start = np.random.randint(start, end - length)
                mask_indices.extend(span_start + j for j in range(length))

                new_parts = []
                if span_start - start - min_space >= keep_length:
                    new_parts.append((start, span_start - min_space + 1))
                if end - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, end))
                return new_parts

            parts = [(0, sequence_length)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lengths_in_parts = np.fromiter(
                    (end - start if end - start >= length + min_space else 0 for start, end in parts),
                    int,
                )
                lengths_sum = np.sum(lengths_in_parts)
                if lengths_sum == 0:
                    break
                probabilities = lengths_in_parts / np.sum(lengths_in_parts)
                choice = np.random.choice(len(parts), p=probabilities)
                start, end = parts.pop(choice)
                parts.extend(arrange(start, end, length, min_length))
            mask_indices = np.asarray(mask_indices)
        else:
            min_length = min(lengths)
            if sequence_length - min_length <= number_of_masks:
                min_length = sequence_length - number_of_masks - 1

            mask_indices = np.random.choice(sequence_length - min_length, number_of_masks, replace=False)
            mask_indices = np.asarray(
                [
                    mask_indices[j] + offset
                    for j in range(len(mask_indices))
                    for offset in range(lengths[j])
                ]
            )
        mask_indices = np.unique(mask_indices[mask_indices < sequence_length])

        # Ensure that the first index is never masked
        mask_indices = mask_indices[mask_indices != 0]

        # Ensure that max_mask_percentage_per_window is respected with sliding window
        if max_mask_percentage_per_window > 0 and max_mask_percentage_window_size > 0:
            max_masks_per_window = int(max_mask_percentage_window_size * max_mask_percentage_per_window)
            for start in range(sequence_length - max_mask_percentage_window_size + 1):
                end = start + max_mask_percentage_window_size
                window_mask_indices = mask_indices[(mask_indices >= start) & (mask_indices < end)]
                if len(window_mask_indices) > max_masks_per_window:
                    excess_count = len(window_mask_indices) - max_masks_per_window
                    excess_indices = np.random.choice(window_mask_indices, excess_count, replace=False)
                    mask_indices = np.setdiff1d(mask_indices, excess_indices)
                    mask_indices = np.sort(mask_indices)

        mask[i, mask_indices] = True

    return mask


class ConformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 ffn_dim,
                 num_heads,
                 num_layers,
                 depthwise_conv_kernel_size: int = 31,
                 dropout=0.,
                 use_group_norm=False,
                 convolution_first=False,
                 left_context: int = 0):
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
                    causal=True,
                    left_context=left_context,
                    right_context=0,
                    attention_type="RelPosMHAXL"
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, padding_mask, output_hidden_states=False, output_attentions=False):
        pos_embs = self.positional_encoder(x)
        hidden_states = []
        attentions = []
        for layer in self.conformer_layers:
            x, attention = layer(x, padding_mask, pos_embs)
            hidden_states.append(x)
            attentions.append(attention)
        output = (x,)
        if output_hidden_states:
            output += (hidden_states,)
        if output_attentions:
            output += (attentions,)
        return output[0] if len(output) == 1 else output


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0., projection_size=0, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            proj_size=projection_size,
                            **kwargs)

    def forward(self, x, padding_mask, hidden=None, output_hidden: bool = False):
        lengths = _padding_mask_to_lengths(padding_mask)
        # first pack the padded sequences
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # lstm pass
        out_packed, hidden = self.lstm(x_packed, hidden)
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)
        output = (out_padded, lengths)
        if output_hidden:
            output += (hidden,)

        return output


class xLSTM(nn.Module):
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


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, transpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.transpose_dim = transpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.transpose_dim, -1)


class DecoderLayer(nn.Module):
    def __init__(self, input_embedding_dim, decoder_embedding_dim, kernel_size, groups):
        super().__init__()
        self.input_embedding_dim = input_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.conv = nn.Conv1d(in_channels=input_embedding_dim,
                              out_channels=decoder_embedding_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size - 1,
                              groups=groups)
        self.pad = SamePad(kernel_size, causal=True)
        self.transpose_last1 = TransposeLast()
        self.layer_norm = nn.LayerNorm(normalized_shape=decoder_embedding_dim, elementwise_affine=False)
        self.transpose_last2 = TransposeLast()
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.pad(x)
        x = self.transpose_last1(x)
        x = self.layer_norm(x)
        x = self.transpose_last2(x)
        x = self.gelu(x)
        x = x + residual if self.input_embedding_dim == self.decoder_embedding_dim else x
        return x


class Decoder(nn.Module):
    def __init__(self, input_embedding_dim, decoder_embedding_dim, kernel_size, groups, decoder_layers,
                 projection_layers):
        super().__init__()
        blocks = []
        for i in range(decoder_layers - 1):
            blocks.append(DecoderLayer(input_embedding_dim if i == 0 else decoder_embedding_dim,
                                       decoder_embedding_dim,
                                       kernel_size,
                                       groups))
        self.blocks = nn.Sequential(*blocks)
        projections = []
        curr_dim = decoder_embedding_dim
        for i in range(projection_layers - 1):
            next_dim = int(curr_dim * 2) if i == 0 else curr_dim
            projections.append(nn.Linear(curr_dim, next_dim))
            projections.append(nn.GELU())
            curr_dim = next_dim
        projections.append(nn.Linear(curr_dim, input_embedding_dim))
        if len(projections) == 1:
            self.projections = projections[0]
        else:
            self.projections = nn.Sequential(*projections)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.blocks(x)
        x = x.transpose(1, 2)
        x = self.projections(x)
        return x


@dataclass
class LogMelConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    n_fft: int = 400
    window_length: int = 400
    hop_length: int = 160
    stacked_consecutive_features: int = 1
    stacked_features_stride: int = 1


@dataclass
class CNNExtractorConfig:
    in_channels: Tuple[int, ...] = (1, 512, 512, 512, 512, 512, 512)
    out_channels: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    strides: Tuple[int, ...] = (5, 3, 2, 2, 2, 2, 2)
    stacked_consecutive_features: int = 1
    stacked_features_stride: int = 1


@dataclass
class FeatureExtractorConfig:
    type: str = "cnn"
    config: Union[CNNExtractorConfig, LogMelConfig] = field(default_factory=CNNExtractorConfig)


def initialize_feature_extractor(config: FeatureExtractorConfig):
    if config.type.lower() == "logmel":
        return LogMel(**config.config)
    elif config.type.lower() == "cnn":
        return CNNFeatureExtractor(**config.config)
    else:
        raise ValueError(f"Unknown feature extractor type: {config.type}")


@dataclass
class LSTMEncoderConfig:
    input_dim: int = 40
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.


@dataclass
class ConformerEncoderConfig:
    in_channels: Tuple[int, ...] = (1, 512, 512, 512, 512, 512, 512)
    out_channels: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    strides: Tuple[int, ...] = (5, 3, 2, 2, 2, 2, 2)
    stacked_consecutive_features: int = 1
    stacked_features_stride: int = 1


@dataclass
class xLSTMEncoderConfig:
    in_channels: Tuple[int, ...] = (1, 512, 512, 512, 512, 512, 512)
    out_channels: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    strides: Tuple[int, ...] = (5, 3, 2, 2, 2, 2, 2)
    stacked_consecutive_features: int = 1
    stacked_features_stride: int = 1


@dataclass
class EncoderConfig:
    type: str = "Conformer"
    config: Union[LSTMEncoderConfig, ConformerEncoderConfig, xLSTMEncoderConfig] \
        = field(default_factory=ConformerEncoderConfig)


def initialize_encoder(config: EncoderConfig):
    if config.type.lower() == "conformer":
        return ConformerEncoder(**config.config)
    elif config.type.lower() == "lstm":
        return nn.LSTM(**config.config)
    else:
        raise ValueError(f"Unknown encoder type: {config.type}")


@dataclass
class BPCConfig:
    input_dim: int = 512
    input_dropout: float = 0.0
    encoder_num_heads: int = 8
    encoder_num_layers: int = 4
    encoder_embedding_dim: int = 512
    depthwise_conv_kernel_size: int = 31
    encoder_left_context: int = 31
    decoder_layers: int = 5
    decoder_kernel_size: int = 5
    decoder_groups: int = 16
    decoder_embedding_dim: int = 512
    projection_layers: int = 1
    average_top_k_layers: int = 8
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    sample_rate: int = 16000
    mask_prob: float = 0.65
    mask_length: int = 10
    mask_selection: str = 'static'
    mask_other: float = 0.0
    no_mask_overlap: bool = False
    mask_min_space: int = 0
    max_mask_percentage_per_window: float = 0.0
    max_mask_percentage_window_size: int = 0
    mask_channel_prob: float = 0.0
    mask_channel_selection: str = 'static'
    mask_channel_other: float = 0.0
    mask_channel_length: int = 10
    no_mask_channel_overlap: bool = False
    mask_channel_min_space: int = 0
    time_shift: int = 3
    ema_decay: float = 0.999
    ema_end_decay: float = 0.9999
    ema_anneal_end_step: float = 20000

    def update(self, cfg: dict):
        for key, value in cfg.items():
            if key == "feature_extractor" and isinstance(value, dict):
                config_classes = {
                    "cnn": CNNExtractorConfig,
                    "log_mel": LogMelConfig
                }
                self.feature_extractor.type = value['type']
                self.feature_extractor.config = config_classes[self.feature_extractor.type](**value["config"])

            elif hasattr(self, key):
                setattr(self, key, value)


class BPC(nn.Module):
    """
    Bootstrap Predictive Coding main module.
    """

    def __init__(self,
                 cfg: BPCConfig = None):
        super().__init__()
        self.feature_extractor = initialize_feature_extractor(cfg.feature_extractor)
        self.input_dropout = nn.Dropout(cfg.input_dropout)
        self.input_projection = nn.Linear(cfg.input_dim, cfg.encoder_embedding_dim) \
            if cfg.input_dim != cfg.encoder_embedding_dim else None
        self.input_dim = cfg.input_dim
        self.encoder = ConformerEncoder(input_dim=cfg.encoder_embedding_dim,
                                        ffn_dim=cfg.encoder_embedding_dim * cfg.encoder_num_heads,
                                        num_layers=cfg.encoder_num_layers,
                                        num_heads=cfg.encoder_num_heads,
                                        depthwise_conv_kernel_size=cfg.depthwise_conv_kernel_size,
                                        left_context=cfg.encoder_left_context)
        self.ema = EMA(self.encoder)  # Instantiate teacher encoder
        self.decoder_layers = cfg.decoder_layers
        self.decoder_kernel_size = cfg.decoder_kernel_size
        self.decoder_groups = cfg.decoder_groups
        self.encoder_embedding_dim = cfg.encoder_embedding_dim
        self.decoder_embedding_dim = cfg.decoder_embedding_dim
        self.projection_layers = cfg.projection_layers
        self.decoder = Decoder(cfg.encoder_embedding_dim, cfg.decoder_embedding_dim, cfg.decoder_kernel_size,
                               cfg.decoder_groups,
                               cfg.decoder_layers, cfg.projection_layers)
        self.average_top_k_layers = cfg.average_top_k_layers
        self.mask_prob = cfg.mask_prob
        self.mask_length = cfg.mask_length
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        self.max_mask_percentage_per_window = cfg.max_mask_percentage_per_window
        self.max_mask_percentage_window_size = cfg.max_mask_percentage_window_size
        self.ema_decay = cfg.ema_decay
        self.ema_end_decay = cfg.ema_end_decay
        self.ema_anneal_end_step = cfg.ema_anneal_end_step

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(self.encoder_embedding_dim).uniform_()
        )
        self.time_shift = cfg.time_shift

    def make_targets(self, target_layer_results, num_layers):
        target_layer_results = target_layer_results[-num_layers:]
        target_layer_results = [
            tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
        ]
        target_layer_results = [
            F.instance_norm(tl.float()) for tl in target_layer_results
        ]
        target_layer_results = [
            tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
        ]
        targets = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            targets.add_(tl.float())
        targets = targets.div_(len(target_layer_results))
        targets = F.instance_norm(targets.transpose(1, 2)).transpose(1, 2)
        return targets

    def apply_mask(self, x, padding_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                shape=(B, T),
                padding_mask=padding_mask,
                mask_probability=self.mask_prob,
                mask_length=self.mask_length,
                mask_type=self.mask_selection,
                mask_other=self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
                max_mask_percentage_per_window=self.max_mask_percentage_per_window,
                max_mask_percentage_window_size=self.max_mask_percentage_window_size
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                shape=(B, C),
                padding_mask=None,
                mask_probability=self.mask_channel_prob,
                mask_length=self.mask_channel_length,
                mask_type=self.mask_channel_selection,
                mask_other=self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(self, teacher_input, student_input: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None, mask: Optional[bool] = None,
                extract_features: Optional[bool] = False):

        padding_mask = _lengths_to_padding_mask(lengths)

        teacher_features = self.feature_extractor(teacher_input).transpose(1, 2)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(teacher_features, padding_mask)

        if self.input_projection:
            teacher_features = self.input_projection(teacher_features)

        teacher_features = self.input_dropout(teacher_features)

        with torch.no_grad():
            self.ema.model.eval()
            _, teacher_hidden_states = self.ema.model(teacher_features, padding_mask, output_hidden_states=True)

        if extract_features:
            return teacher_hidden_states

        target = self.make_targets(teacher_hidden_states, self.average_top_k_layers)

        if student_input is None:
            student_features = teacher_features
        else:
            student_features = self.feature_extractor(student_input).transpose(1, 2)
            if self.input_projection:
                student_features = self.input_projection(student_features)

        if mask:
            student_features, masked_indices = self.apply_mask(
                student_features, padding_mask
            )
        else:
            masked_indices = None

        if self.time_shift:
            student_features = student_features[:, :-self.time_shift]
            target = target[:, self.time_shift:]
            padding_mask = padding_mask[:, :-self.time_shift]
            if mask:
                masked_indices = masked_indices[:, :-self.time_shift]

        prediction = self.encoder(student_features, padding_mask)
        prediction = self.decoder(prediction)

        if masked_indices is not None:
            prediction = prediction[masked_indices]
            target = target[masked_indices]
        return prediction, target

    def ema_step(self):
        """
        Function which to step the EMA encoder
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)


if __name__ == '__main__':
    bpc_cfg = BPCConfig()
    model = BPC(bpc_cfg)

    SR = 16000
    T = 3

    source = torch.randn(4, SR * T)
    lengths = torch.Tensor([s.size(-1) for s in source])

    prediction_out, target_out = model(source, lengths, mask=True)
    print(prediction_out.shape)
