from typing import Tuple, List
from math import prod

import torch
from torch import nn
from torch.nn import functional as F


def stack_consecutive_features(features: torch.Tensor, n_consecutive_features, stride):
    # check if padding is needed
    padding = (features.size(-1) - n_consecutive_features) % stride
    if padding:
        features = F.pad(features, (0, padding))
    features = features.unfold(dimension=-1, size=n_consecutive_features, step=stride)
    features = features.transpose(1, 2).transpose(-1, -2)
    features = features.reshape(features.size(0), features.size(1), -1).transpose(-1, -2)
    return features


def build_cnn_feature_extractor(
        in_channels: Tuple[int, ...] = (40, 512, 512, 512, 512, 512, 512),
        out_channels: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512),
        kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2),
        strides: Tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2),
        causal: bool = False
) -> nn.Module:
    """
    Builds a CNN feature extractor with an option for causal or non-causal convolutions.

    Parameters:
    -----------
    in_channels : Tuple[int, ...]
        A tuple representing the number of input channels for each convolutional layer.
    out_channels : Tuple[int, ...]
        A tuple representing the number of output channels for each convolutional layer.
    kernel_sizes : Tuple[int, ...]
        A tuple representing the kernel sizes for each convolutional layer.
    strides : Tuple[int, ...]
        A tuple representing the strides for each convolutional layer.
    causal : bool
        If True, constructs causal convolutions; otherwise, constructs non-causal convolutions.

    Returns:
    --------
    nn.Module
        A sequential container of the CNN layers.
    """
    cnn_blocks: List[nn.Module] = []
    assert (len(in_channels) == len(out_channels) and len(out_channels) == len(kernel_sizes) and
            len(kernel_sizes) == len(strides)), "Inconsistent number of layer parameters"

    for i, _ in enumerate(in_channels):
        if causal:
            padding = (kernel_sizes[i] - 1) if i == 0 else (kernel_sizes[i] - strides[i])
        else:
            padding = (kernel_sizes[i] - 1) // 2

        conv_layer = nn.Conv1d(in_channels=in_channels[i],
                               out_channels=out_channels[i],
                               kernel_size=kernel_sizes[i],
                               stride=strides[i],
                               padding=padding,
                               bias=False)
        if causal:
            conv_layer.padding_mode = 'zeros'  # Ensure the padding is zero for causality

        cnn_blocks.append(conv_layer)

        if i == 1:
            cnn_blocks.append(nn.GroupNorm(out_channels[i], out_channels[i]))
        cnn_blocks.append(nn.GELU())

    return nn.Sequential(*cnn_blocks)


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels: Tuple[int, ...] = (1, 512, 512, 512, 512, 512, 512),
                 out_channels: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512),
                 kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2),
                 strides: Tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2),
                 stacked_consecutive_features: int = 1,
                 stacked_features_stride: int = 1,
                 causal: bool = False,
                 sample_rate : int = 16000):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.sample_rate = sample_rate
        self.causal = causal
        self.cnn = build_cnn_feature_extractor(in_channels=in_channels, out_channels=out_channels,
                                               kernel_sizes=kernel_sizes, strides=strides, causal=causal)
        self.stacked_consecutive_features = stacked_consecutive_features
        self.stacked_features_stride = stacked_features_stride
        self.feature_rate = self._compute_feature_rate()
        self.receptive_field = self._compute_receptive_field()

    def _compute_receptive_field(self):
        """
        Compute the receptive field for features extracted from an N layer CNN.

        Parameters:
        kernel_sizes (list of int): A list of kernel sizes for each layer.
        strides (list of int): A list of stride sizes for each layer.

        Returns:
        receptive_fields (float): The receptive fields of the output from the CNN layers (in seconds).
        """
        if len(self.kernel_sizes) != len(self.strides):
            raise ValueError("The length of kernel_sizes and strides must be the same")

        num_layers = len(self.kernel_sizes)

        # Initialize the receptive field for the first layer
        receptive_fields = [self.kernel_sizes[0]]
        # Compute the receptive field for each subsequent layer
        jump = 1
        for i in range(1, num_layers):
            jump = jump * self.strides[i-1]
            receptive_fields.append(receptive_fields[i - 1] + (self.kernel_sizes[i] - 1) * jump)

        return receptive_fields[-1] / self.sample_rate

    def _compute_feature_rate(self):
        return round(self.sample_rate / (prod(self.strides) * self.stacked_features_stride))

    def forward(self, waveform: torch.Tensor):
        features = self.cnn(waveform)
        if self.stacked_consecutive_features > 1:
            features = stack_consecutive_features(features,
                                                  n_consecutive_features=self.stacked_consecutive_features,
                                                  stride=self.stacked_features_stride)
        return features
