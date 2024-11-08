from typing import Optional
from torch.utils.data import DataLoader, Sampler, default_collate


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=default_collate,
                    pin_memory: bool = False, sampler: Optional[Sampler] = None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, sampler=sampler)
    return data_loader
