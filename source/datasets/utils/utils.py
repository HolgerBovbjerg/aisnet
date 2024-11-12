import os
from logging import getLogger
from typing import Optional

import yaml
from torch.utils.data import DataLoader, Sampler, default_collate


logger = getLogger(__name__)

def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=default_collate,
                    pin_memory: bool = False, sampler: Optional[Sampler] = None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, sampler=sampler)
    return data_loader

def load_database():
    database_path = "data/database.yaml"

    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file '{database_path}' not found.")

    with open(database_path, "r", encoding="utf8") as f:
        database = yaml.safe_load(f)

    return database

def check_database(dataset_name: str):
    database = load_database()
    path = database.get(dataset_name, None)
    if path is None:
        logger.error("Dataset '%s' not found in database. "
                     "Please update database.yaml. "
                     "Set '%s: download' if %s should be downloaded "
                     "or '%s: /path/to/%s' if %s can be accessed via path",
                     dataset_name, dataset_name, dataset_name, dataset_name, dataset_name, dataset_name)
    return path