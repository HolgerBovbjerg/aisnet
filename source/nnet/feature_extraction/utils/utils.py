from typing import Union, Tuple

import numpy as np


def hz_to_mel(hz: float) -> Union[float, np.ndarray]:
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 700 * (10 ** (mel / 2595.0) - 1)

