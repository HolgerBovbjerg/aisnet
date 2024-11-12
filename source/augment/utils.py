from torch_audiomentations import Compose

from .specaugment import SpecAugment
from .add_rir import AddRIR
from .add_noise import AddNoise


def get_augmentor(name, **kwargs):
    augmentor_classes = {
        "rir": AddRIR,
        "noise": AddNoise,
        "specaugment": SpecAugment
    }

    if name in augmentor_classes:
        return augmentor_classes[name](**kwargs)
    raise ValueError(f"Unknown augmentor name: {name}")


class ComposedAugmentorWrapper:
    def __init__(self, composed_augmentors):
        self.composed_augmentors = composed_augmentors

    def __call__(self, sample):
        return self.composed_augmentors(sample).samples


def get_composed_augmentations(config, sampling_rate: int = 16000):
    augmentations = [
        get_augmentor(key, sampling_rate=sampling_rate, **config[key])
        for key in config.keys()
    ]
    return ComposedAugmentorWrapper(Compose(augmentations, output_type="dict"))
