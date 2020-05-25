from typing import Tuple, Union

import numpy as np

from .format import WaveFormat, FormatTag
from ..signal import Signal, Stereo, Monaural


def normalize(wave_format: WaveFormat, data: np.ndarray, *, threshold: int = 0.99) -> np.ndarray:
    if wave_format.format_code == FormatTag.ieee_float or wave_format.format_code == FormatTag.pcm:
        max_sample = np.abs(data).max()
        return data / max_sample if max_sample < threshold else data
    else:
        raise NotImplementedError


def monauralize(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1)


def wave_to_signal(wave_format: WaveFormat, data: np.ndarray) -> Union[Signal, Monaural, Stereo]:
    if wave_format.number_of_channels == 1:
        return Monaural(n_samples=len(data), sampling_rate=wave_format.sampling_rate, data=data.T)
    elif wave_format.number_of_channels == 2:
        return Stereo(n_samples=len(data), sampling_rate=wave_format.sampling_rate, data=data.T)
    else:
        return Signal(n_samples=len(data),
                      n_channels=wave_format.number_of_channels,
                      sampling_rate=wave_format.sampling_rate,
                      data=data.T)


def signal_to_wave(signal: Signal) -> Tuple[WaveFormat, np.ndarray]:
    wf: WaveFormat = WaveFormat.create_default(n_channels=signal.number_of_channels, sampling_rate=signal.sampling_rate)
    data: np.ndarray = np.array(signal.raw.T, dtype="<f4", order="C")
    return wf, data
