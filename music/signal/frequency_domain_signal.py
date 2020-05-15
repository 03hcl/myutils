from typing import Optional, Tuple

import numpy as np

from .exceptions import WindowSizeError


class FrequencyDomainSignalBase:

    # region Property

    @property
    def number_of_channels(self) -> int:
        return self._raw.shape[0]

    @property
    def number_of_frequencies(self) -> int:
        return self._frequencies.shape[0]   # self._raw.shape[2]

    @property
    def number_of_samples(self) -> int:
        return self._seconds.shape[0]       # self._raw.shape[1]

    @property
    def sampling_interval(self) -> float:
        return self._sampling_interval

    @property
    def sampling_rate(self) -> float:
        return 1 / self._sampling_interval

    @property
    def raw(self) -> np.ndarray:
        return self._raw

    @property
    def frequencies(self) -> np.ndarray:
        return self._frequencies

    @property
    def seconds(self) -> np.ndarray:
        return self._seconds

    # endregion

    def __init__(self, data: np.ndarray, frequencies: np.ndarray, sampling_interval: float, seconds: np.ndarray,
                 *args, **kwargs):
        self._raw: np.ndarray = data
        self._frequencies: np.ndarray = frequencies
        self._seconds: np.ndarray = seconds
        self._sampling_interval: float = sampling_interval

    def view_moving_window(self, window_size: int,
                           *, stride: int = 1, pad_width: Optional[Tuple[int, int]] = (0, 0)) -> np.ndarray:
        if window_size > self.number_of_samples:
            raise WindowSizeError
        stride: int = FrequencyDomainSignalBase._get_stride(window_size, stride)
        if pad_width is None:
            pad_left: int = window_size // 2
            pad_width = (pad_left, window_size - pad_left)
        padded: np.ndarray = self._raw if pad_width == (0, 0) \
            else np.pad(self._raw, [(0, 0), pad_width, (0, 0)], mode="constant").astype(dtype="<c8", order="C")
        data_size = int((padded.shape[1] - window_size) / stride) + 1
        return np.lib.stride_tricks.as_strided(padded,
                                               shape=(padded.shape[0],
                                                      data_size,
                                                      window_size,
                                                      self.number_of_frequencies),
                                               strides=(padded.strides[0],
                                                        padded.strides[1] * stride,
                                                        padded.strides[1],
                                                        padded.strides[2]))

    @classmethod
    def _get_stride(cls, window_size: int, stride: int) -> int:
        return stride if stride > 0 else window_size
