import torchaudio
from torch_audiomentations import Compose

from .specaugment import SpecAugment
from .add_rir import AddRIR
from .add_noise import AddNoise


def get_augmentor(name, **kwargs):
    if name == "rir":
        return AddRIR(**kwargs)
    if name == "noise":
        return AddNoise(**kwargs)
    if name == "specaugment":
        return SpecAugment(**kwargs)


def get_composed_augmentations(config, sampling_rate: int = 16000):
    augmentations = []
    for key in config.keys():
        augmentations.append(get_augmentor(key, sampling_rate=sampling_rate, **config[key]))
    return Compose(augmentations)
