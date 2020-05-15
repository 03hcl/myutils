import numpy as np

from .frequency_domain_signal import FrequencyDomainSignalBase


class FFT(FrequencyDomainSignalBase):

    @property
    def frequency_stride(self) -> float:
        return self._frequency_stride

    @property
    def window_function(self) -> np.ndarray:
        return self._window_function

    def __init__(self, data: np.ndarray, frequencies: np.ndarray, sampling_interval: float, seconds: np.ndarray,
                 frequency_stride: float, window_function: np.ndarray, *args, **kwargs):
        super(FFT, self).__init__(data, frequencies, sampling_interval, seconds, *args, **kwargs)
        self._frequency_stride: float = frequency_stride
        self._window_function: np.ndarray = window_function
