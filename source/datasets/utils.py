from typing import Optional

import torch
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    waveforms = [data[0].T for data in batch[0]]
    waveforms = [(waveform - waveform.mean()) / (waveform.std() + 1.e-9) for waveform in waveforms]  # Instance norm

    lengths = torch.tensor([waveform.size(0) for waveform in waveforms])

    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0)

    return waveforms_padded, lengths


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=pad_collate,
                    pin_memory: bool = False, sampler: Optional[Sampler] = None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
                             sampler=sampler)
    return data_loader