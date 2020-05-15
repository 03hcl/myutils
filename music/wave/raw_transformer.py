import numpy as np

from .format import WaveFormat, FormatTag


def normalize(wave_format: WaveFormat, data: np.ndarray, *, threshold: int = 0.99) -> np.ndarray:
    if wave_format.format_code == FormatTag.ieee_float:
        max_sample = np.abs(data).max()
        return data / max_sample if max_sample < threshold else data
    else:
        raise NotImplementedError


def monauralize(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1)


