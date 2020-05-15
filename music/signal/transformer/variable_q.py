from typing import Optional

from .constant_q import ConstantQ
from .window_function import WindowFunction


class VariableQ(ConstantQ):

    @property
    def gamma(self) -> float:
        return self._gamma

    def __init__(self, *, f_min: float = 440 / 16, f_max: float = 440 * 32, b: int = 24, gamma: Optional[float] = None,
                 sampling_rate: int = 48000, window_size_rate: float = 0,
                 window_type: WindowFunction = WindowFunction.Hamming, zero_threshold: float = 0.0015,
                 **kwargs):

        super(VariableQ, self).__init__(f_min=f_min, f_max=f_max, b=b,
                                        sampling_rate=sampling_rate, window_size_rate=window_size_rate,
                                        window_type=window_type, zero_threshold=zero_threshold,
                                        _precreate_kernel=False, **kwargs)

        self._window_size_rate: float = 1 if window_size_rate <= 0 else window_size_rate

        self._Q: float = \
            1 / ((2 ** (1 / self._number_of_bins_in_an_octave)) - (2 ** (-1 / self._number_of_bins_in_an_octave)))

        if gamma is None:
            self._gamma: float = 24.7 * 9.265 / self._Q if gamma is None else gamma
        else:
            self._gamma: float = gamma

        self._set_window_size_and_kernel(window_type)

    def n_k(self, f_k: float) -> int:
        return int(self._window_size_rate * self._sampling_rate / (f_k / self._Q + self._gamma))

    def _get_q(self, k: int) -> float:
        return 1 / (1 / self._Q + self._gamma / self.f_k(k)) * self._window_size_rate
