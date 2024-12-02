from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Converts lengths into a padding mask tensor.

    Args:
        lengths (torch.Tensor): A 1D tensor of shape (batch_size,) containing the lengths for each sequence.

    Returns:
        torch.Tensor: A 2D boolean tensor of shape (batch_size, max_length) where `True` indicates padding positions.
    """
    batch_size = lengths.size(0)
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


def padding_mask_to_lengths(padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Converts a padding mask back to lengths.

    Args:
        padding_mask (torch.Tensor): A 2D boolean tensor of shape (batch_size, max_length)
                                     where `True` indicates padding positions.

    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) containing the lengths for each sequence.
    """
    # Calculate the lengths by summing the non-padding positions (where padding_mask is False)
    lengths = (~padding_mask).sum(dim=1)
    return lengths


def generate_attention_mask(size: tuple, left_context: int, right_context: Optional[int] = None, device: str = None) -> torch.Tensor:
    """
    Generates an attention mask tensor based on the specified left and right context sizes.

    The attention mask is a binary mask where:
    - `True` values indicate positions that should be masked (i.e., not attended to).
    - `False` values indicate positions that should not be masked (i.e., should be attended to).

    The mask is created by considering both left and right context around each position.
    If `right_context` is not provided, it is set to the length of the input (first element of `size`).
    If `left_context` is not provided, it is set to the length of the input as well.

    Args:
        size (tuple): A tuple of two integers (length, length), specifying the size of the square attention mask.
        left_context (int): The number of positions to the left of the current position that should be attended to.
        right_context (Optional[int], optional): The number of positions to the right of the current position that should be attended to. If not provided, defaults to `size[0]`.
        device (str, optional): The device on which to create the tensor (e.g., 'cpu' or 'cuda'). Defaults to `None`.

    Returns:
        torch.Tensor: A binary attention mask of shape `size`, with `True` values indicating positions that should be masked, and `False` values indicating positions that should not be masked.
    """
    if right_context is None:
        right_context = size[0]
    if left_context is None:
        left_context = size[0]
    mask = ~torch.triu(torch.tril(torch.ones(size=size, device=device),
                                  diagonal=right_context),
            diagonal=-left_context).bool()
    return mask


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
            number of time steps divided by length of mask span to mask approximately this percentage of all elements.
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


@dataclass
class MaskGeneratorConfig:
    mask_prob: float = 0.65
    mask_length: int = 10
    mask_selection: str = 'static'
    min_masks: int = 2
    mask_other: float = 0.0
    no_mask_overlap: bool = False
    mask_min_space: int = 0
    max_mask_percentage_per_window: float = 0.0
    max_mask_percentage_window_size: int = 0


class MaskGenerator(nn.Module):
    def __init__(self, cfg: MaskGeneratorConfig):
        super().__init__()
        self.mask_prob = cfg.mask_prob
        self.mask_length = cfg.mask_length
        self.mask_selection = cfg.mask_selection
        self.min_masks = cfg.min_masks
        self.mask_other = cfg.mask_other
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.max_mask_percentage_per_window = cfg.max_mask_percentage_per_window
        self.max_mask_percentage_window_size = cfg.max_mask_percentage_window_size

    def generate_mask(self, x, lengths: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        padding_mask = lengths_to_padding_mask(lengths)
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                shape=(B, T),
                padding_mask=padding_mask,
                mask_probability=self.mask_prob,
                mask_length=self.mask_length,
                mask_type=self.mask_selection,
                mask_other=self.mask_other,
                min_masks=self.min_masks,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
                max_mask_percentage_per_window=self.max_mask_percentage_per_window,
                max_mask_percentage_window_size=self.max_mask_percentage_window_size
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
        else:
            mask_indices = None
        return mask_indices
