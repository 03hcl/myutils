import math
import numpy as np
from numpy import log2
from scipy.sparse import csr_matrix

from .window_function import WindowFunction


class ConstantQ:

    # region Property

    @property
    def f_min(self) -> float:
        return self._f_min

    @property
    def f_max(self) -> float:
        return self._f_max

    @property
    def number_of_bins_in_an_octave(self) -> int:
        return self._number_of_bins_in_an_octave

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def window_size_rate(self) -> float:
        return self._window_size_rate

    @property
    def zero_threshold(self) -> float:
        return self._zero_threshold

    @property
    def q(self) -> float:
        return self._Q

    @property
    def n(self) -> int:
        return self._N

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def kernel(self) -> csr_matrix:
        return self._kernel

    # endregion

    def __init__(self, *, f_min: float = 440 / 16, f_max: float = 440 * 32, b: int = 24,
                 sampling_rate: int = 48000, window_size_rate: float = 0,
                 window_type: WindowFunction = WindowFunction.Hamming, zero_threshold: float = 0.0015,
                 _precreate_kernel: bool = True, **kwargs):

        self._f_min: float = f_min
        self._f_max: float = f_max
        self._number_of_bins_in_an_octave: int = b

        self._sampling_rate: float = sampling_rate
        self._window_size_rate: float = 20 / b if window_size_rate <= 0 else window_size_rate
        self._zero_threshold = zero_threshold

        self._Q: float = (1 / ((2 ** (1 / self._number_of_bins_in_an_octave)) - 1))
        self._N: int = ConstantQ.calculate_n(self._f_min, self._f_max, self._number_of_bins_in_an_octave)
        # self._N: int = int(round(log2(self._f_max / self._f_min) * self._number_of_bins_in_an_octave)) + 1

        # self._f_s: np.ndarray = np.array([self.f_k(k) for k in range(self._N)], dtype="<f4")

        if _precreate_kernel:
            self._set_window_size_and_kernel(window_type)

        super(ConstantQ, self).__init__(**kwargs)

    def f_k(self, k: int) -> float:
        return self._f_min * (2 ** (k / self._number_of_bins_in_an_octave))

    def n_k(self, f_k: float) -> int:
        return int(self._window_size_rate * self._Q * self._sampling_rate / f_k)

    def _set_window_size_and_kernel(self, window_type: WindowFunction) -> None:
        self._window_size: int = 2 ** int(math.ceil(log2(int(self.n_k(self.f_k(0))))))
        self._kernel: csr_matrix = self._create_kernel(window_type)

    def _create_kernel(self, window_type: WindowFunction) -> csr_matrix:

        two_pi_j = 2 * np.pi * 1j

        kernel: np.ndarray = np.zeros((self._N, self._window_size), dtype="<c8")

        for k in range(self._N):
            kernel_k: np.ndarray = np.zeros(self._window_size, dtype="<c8")
            # print(k + 1, "/", self._N)
            f_k: float = self.f_k(k)
            n_k: int = self.n_k(f_k)
            start_window_index: int = (self._window_size - n_k) // 2
            kernel_k[start_window_index: start_window_index + n_k] \
                = (window_type.create(n_k) / n_k) * np.exp(
                two_pi_j * self._get_q(k) * np.arange(n_k, dtype="<f4") / n_k)
            kernel[k, :] = np.fft.fft(kernel_k)

        kernel[abs(kernel) <= self._zero_threshold] = 0

        sparse_kernel = csr_matrix(kernel)
        sparse_kernel = sparse_kernel.conjugate() / self._window_size
        return sparse_kernel

    def _get_q(self, k: int) -> float:
        return self._window_size_rate * self._Q

    @classmethod
    def calculate_n(cls, f_min: float, f_max: float, b: int) -> int:
        return int(round(log2(f_max / f_min) * b)) + 1
