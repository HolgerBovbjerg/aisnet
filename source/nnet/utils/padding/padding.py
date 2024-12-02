import torch


def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask
