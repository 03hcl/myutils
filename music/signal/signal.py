import math
from collections import OrderedDict
from collections.abc import MutableMapping
from itertools import zip_longest
from typing import Callable, Dict, Iterator, List, Optional, TypeVar, Tuple, Union

import numpy as np
from numpy import log2
from numpy.lib.stride_tricks import broadcast_to
from scipy.sparse import csr_matrix

from .fft import FFT
from .transformer import ConstantQ, VariableQ, WindowFunction
from .vqt import VQT
from .exceptions import NotExistKeyTupleError, NotMatchNumberOfChannelsError, WindowSizeError

_signal = TypeVar("_signal", bound="Signal")
_signal_base = TypeVar("_signal_base", bound="SignalBase")
_monaural = TypeVar("_monaural", bound="Monaural")

KeyFromStrDict = Dict[str, int]
KeyDict = Dict[int, str]
# KeyFromStrDict = OrderedDict[str, int]
# KeyDict = OrderedDict[int, str]


class SignalBase(MutableMapping):

    # region Property

    @property
    def number_of_channels(self) -> int:
        return self._number_of_channels

    @property
    def number_of_samples(self) -> int:
        return self._number_of_samples

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def raw(self) -> np.ndarray:
        return self._raw

    @property
    def raw_keys_from_str(self) -> KeyFromStrDict:
        return self._keys_from_str

    @property
    def raw_keys(self) -> KeyDict:
        return self._keys

    @property
    def seconds(self) -> float:
        return float(self._number_of_samples) / self._sampling_rate

    # endregion

    def __init__(self, *,
                 n_channels: int, n_samples: int, sampling_rate: int,
                 keys_from_str: Optional[KeyFromStrDict] = None, keys: Optional[KeyDict] = None,
                 raw: Optional[np.ndarray] = None, **kwargs):

        self._number_of_channels: int = n_channels
        self._number_of_samples: int = n_samples
        self._sampling_rate: int = sampling_rate

        if keys_from_str is not None:
            self._keys_from_str: KeyFromStrDict = keys_from_str
        if keys is not None:
            self._keys: KeyDict = keys
        if raw is not None:
            self._raw: np.ndarray = raw

        super(SignalBase, self).__init__(**kwargs)

    # region Special Method

    def __contains__(self, item: np.ndarray) -> bool:
        raise NotImplementedError

    def __getitem__(self, key: Union[int, str, Tuple[int, str], slice]) -> Union[np.ndarray, _signal]:
        if type(key) == str:
            return self._raw[self._keys_from_str[key]]
        elif type(key) == int:
            return self._raw[key]
        elif type(key) == tuple:
            if key in self._keys.items():
                k: Tuple[int, str] = key
                return self._raw[k[0]]
            else:
                raise NotExistKeyTupleError
        elif type(key) == slice:
            return Signal._create_with_raw(self, self._raw[:, key])
        raise NotImplementedError

    def __setitem__(self, key: Union[int, str, Tuple[int, str], slice], value: np.ndarray) -> None:
        if type(key) == str:
            self._raw[self._keys_from_str[key], :] = value
        elif type(key) == int:
            self._raw[key, :] = value
        elif type(key) == tuple:
            if key in self._keys.items():
                k: Tuple[int, str] = key
                self._raw[k[0], :] = value
            else:
                raise NotExistKeyTupleError
        elif type(key) == slice:
            self._raw[:, key] = value
        else:
            raise NotImplementedError

    def __delitem__(self, key: Union[int, str, Tuple[int, str]]) -> None:
        raise NotImplementedError

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.values())

    def __len__(self) -> int:
        return self.number_of_samples

    # endregion

    # region Mix-in Mapping Method

    # def keys(self) -> List[Tuple[int, str]]:
    #     return [k for k in self._keys.items()]

    # def items(self) -> List[Tuple[Tuple[int, str], np.ndarray]]:
    #     return [(k, v) for (k, v) in zip(self._keys.items(), np.split(self._raw, self.number_of_channels, axis=0))]

    def keys(self) -> Iterator[Tuple[int, str]]:
        for k in self._keys.items():
            yield k

    def items(self) -> Iterator[Tuple[Tuple[int, str], np.ndarray]]:
        for (k, v) in zip(self._keys.items(), np.split(self._raw, self.number_of_channels, axis=0)):
            yield k, v

    def values(self) -> List[np.ndarray]:
        return np.split(self._raw, self.number_of_channels, axis=0)

    def get(self, key: Optional[Union[int, str, Tuple[int, str]]] = None) -> Optional[np.ndarray]:
        if key is None:
            if self.number_of_channels == 1:
                return self._raw[0]
        else:
            if type(key) == str and key in self._keys_from_str.keys():
                return self._raw[self._keys_from_str[key]]
            if type(key) == int and 0 <= key < self.number_of_channels:
                return self._raw[key]
            if type(key) == tuple and key in self._keys.items():
                k: Tuple[int, str] = key
                return self._raw[k[0]]
        return None

    def __eq__(self, other: _signal_base) -> bool:
        return self.raw == other.raw and self.keys() == other.keys()

    def __ne__(self, other: _signal_base) -> bool:
        return not self == other

    # endregion

    # region Mix-in MutableMapping Method (Not Implemented)

    def pop(self, k: Union[int, str, Tuple[int, str]]) -> np.ndarray:
        raise NotImplementedError

    def popitem(self) -> Tuple[Tuple[int, str], np.ndarray]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def update(self, __m, **kwargs) -> None:
        raise NotImplementedError

    def setdefault(self, k: Union[int, str, Tuple[int, str]], default: np.ndarray = ...) -> np.ndarray:
        raise NotImplementedError

    # endregion

    # region Signal Method

    def extract(self, key: Union[int, str, Tuple[int, str], slice]) -> _signal:

        if type(key) == slice:
            return self[key]

        data: np.ndarray = self[key]

        key_name: str = ""

        if type(key) == str:
            key_name = key
        elif type(key) == int:
            key_name = self._keys[key]
        elif type(key) == tuple and key in self._keys.items():
            key_name = key[1]

        return Signal(n_channels=1,
                      n_samples=self.number_of_samples,
                      sampling_rate=self.sampling_rate,
                      channels=[key_name],
                      data=data)

    def monauralize(self) -> _monaural:
        return Monaural(n_samples=self.number_of_samples,
                        sampling_rate=self.sampling_rate,
                        data=self._raw.mean(axis=0))

    def normalize(self, *, threshold: int = 0.99) -> _signal:
        max_sample = np.abs(self.raw).max()
        return Signal._create_with_raw(self, self.raw / max_sample if max_sample < threshold else self.raw)

    def reverse(self) -> _signal:
        return Signal._create_with_raw(self, np.flip(self.raw, axis=1))

    def clip(self) -> _signal:
        return Signal._create_with_raw(self, Signal._clip_raw(self.raw))

    def scale_with_clipping(self, scale: float = 1) -> _signal:
        return Signal._create_with_raw(self, Signal._clip_raw(self.raw * scale))

    def pad_zeros(self, pad_width: Tuple[int, int] = (0, 0)) -> _signal:
        data: np.ndarray = np.pad(self._raw[:, :], [(0, 0), pad_width], mode="constant").astype(dtype="<f4", order="C")
        return Signal._create_with_raw(self, data)

    def view_moving_window(self, window_size: int,
                           *, stride: int = 0, pad_width: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if window_size > self.number_of_samples:
            raise WindowSizeError
        stride: int = SignalBase._get_stride(window_size, stride)
        if pad_width is None:
            pad_left: int = window_size // 2
            pad_width = (pad_left, window_size - pad_left)
        padded: Signal = self if pad_width == (0, 0) else self.pad_zeros(pad_width)
        data_size = int((padded.number_of_samples - window_size) / stride) + 1
        return np.lib.stride_tricks.as_strided(padded.raw,
                                               shape=(padded.raw.shape[0], data_size, window_size),
                                               strides=(padded.raw.strides[0],
                                                        padded.raw.strides[1] * stride,
                                                        padded.raw.strides[1]))

    def fft(self, window_size: int, window_type: WindowFunction = WindowFunction.Hamming,
            *, stride: int = 1) -> FFT:

        if window_size & (window_size - 1):
            raise WindowSizeError
        stride: int = SignalBase._get_stride(window_size, stride)
        moving_window: np.ndarray = self.view_moving_window(window_size, stride=stride)
        window_func: np.ndarray = SignalBase._create_window_function(window_type, moving_window.shape)
        result: np.ndarray = np.fft.rfft(moving_window * window_func)[:, :, 1:] / (window_size / 2)
        freq_interval: int = window_size // 2
        freqs: np.ndarray = np.linspace(float(self.sampling_rate) / freq_interval, self.sampling_rate, freq_interval)
        return FFT(result, freqs, *self._seconds(stride), freq_interval, window_func)
        # return result, self._seconds(stride), freqs

    def create_constant_q(self, window_type: WindowFunction = WindowFunction.Hamming,
                          *, f_min: float = 440 / 16, f_max: float = 440 * 32, b: int = 24,
                          window_size_rate: float = 1, zero_threshold: float = 0.01) -> ConstantQ:
        return ConstantQ(f_min=f_min, f_max=f_max, b=b,
                         sampling_rate=self.sampling_rate, window_size_rate=window_size_rate,
                         window_type=window_type, zero_threshold=zero_threshold)

    def create_variable_q(self, window_type: WindowFunction = WindowFunction.Hamming,
                          *, f_min: float = 440 / 16, f_max: float = 440 * 32, b: int = 24,
                          gamma: Optional[float] = None,
                          window_size_rate: float = 1, zero_threshold: float = 0.01) -> VariableQ:
        return VariableQ(f_min=f_min, f_max=f_max, b=b, gamma=gamma,
                         sampling_rate=self.sampling_rate, window_size_rate=window_size_rate,
                         window_type=window_type, zero_threshold=zero_threshold)

    def cqt(self, window_type: WindowFunction = WindowFunction.Hamming,
            *, stride: int = 1, precalculated_cq: Optional[ConstantQ] = None,
            f_min: float = 440 / 16, f_max: float = 440 * 32, b: int = 24,
            window_size_rate: float = 1, zero_threshold: float = 0.01) -> VQT:
        cq: ConstantQ \
            = precalculated_cq or ConstantQ(f_min=f_min, f_max=f_max, b=b,
                                            sampling_rate=self.sampling_rate, window_size_rate=window_size_rate,
                                            window_type=window_type, zero_threshold=zero_threshold)
        return self._run_vqt(stride=stride, cq=cq, f_min=f_min, f_max=f_max)

    def vqt(self, window_type: WindowFunction = WindowFunction.Hamming,
            *, stride: int = 1, precalculated_vq: Optional[VariableQ] = None,
            f_min: float = 440 / 16, f_max: float = 440 * 32, b: int = 24, gamma: Optional[float] = None,
            window_size_rate: float = 1, zero_threshold: float = 0.01) -> VQT:
        vq: VariableQ \
            = precalculated_vq or VariableQ(f_min=f_min, f_max=f_max, b=b,
                                            gamma=gamma or 24.7 * 9.265 * ((2 ** (1 / b)) - (2 ** (-1 / b))) / 2,
                                            sampling_rate=self.sampling_rate, window_size_rate=window_size_rate,
                                            window_type=window_type, zero_threshold=zero_threshold)
        return self._run_vqt(stride=stride, cq=vq, f_min=f_min, f_max=f_max)

    # endregion

    # region Private Method

    def _seconds(self, stride: int) -> Tuple[float, np.ndarray]:
        sampling_interval: float = stride / self.sampling_rate
        seconds: np.ndarray = np.arange(0, (self.number_of_samples + 1) / self.sampling_rate, sampling_interval)
        return sampling_interval, seconds

    def _run_vqt(self, stride: int, cq: ConstantQ, f_min: float, f_max: float) -> VQT:
        stride: int = SignalBase._get_stride(cq.window_size, stride)
        moving_window: np.ndarray = self.view_moving_window(cq.window_size, stride=stride)
        fft_array: np.ndarray = np.fft.fft(moving_window)
        result: np.ndarray = (csr_matrix(fft_array.reshape(-1, cq.window_size)) * cq.kernel.T).toarray()
        freqs: np.ndarray = f_min * (2 ** np.linspace(0, log2(f_max / f_min), cq.n))
        sampling_interval, seconds = self._seconds(stride)
        return VQT(result.reshape((self.number_of_channels, -1, cq.n)), freqs, sampling_interval, seconds, cq)

    # endregion

    # region Private Class Method

    @classmethod
    def _clip_raw(cls, raw: np.ndarray) -> np.ndarray:
        raw[raw > 1] = 1
        raw[raw < -1] = -1
        return raw

    @classmethod
    def _create_with_raw(cls, base: _signal_base, data: np.ndarray) -> _signal_base:
        n_channels = data.shape[0]
        if n_channels != base.number_of_channels:
            raise NotMatchNumberOfChannelsError
        n_samples = data.shape[1]
        instance: SignalBase = SignalBase(n_channels=n_channels,
                                          n_samples=n_samples,
                                          sampling_rate=base.sampling_rate,
                                          keys_from_str=base.raw_keys_from_str,
                                          keys=base.raw_keys)
        instance._raw = data
        return instance

    @classmethod
    def _create_window_function(cls, window_type: WindowFunction, shape: Tuple, **kwargs) -> np.ndarray:
        if window_type == WindowFunction.Rectangular:
            return np.ones(shape)
        window_size: int = shape[-1]
        window = window_type.create(window_size, **kwargs).astype(dtype="<f4", order="C")
        return broadcast_to(window, shape)

    @classmethod
    def _get_stride(cls, window_size: int, stride: int) -> int:
        return stride if stride > 0 else window_size

    # endregion


class Signal(SignalBase):

    def __init__(self, *args,
                 n_channels: int, n_samples: int, sampling_rate: int,
                 channels: Optional[List[str]] = None, data: Optional[np.ndarray] = None, **kwargs):

        super(Signal, self).__init__(n_channels=n_channels, n_samples=n_samples, sampling_rate=sampling_rate,
                                     *args, **kwargs)

        if channels is None:
            channels = []
        self._keys_from_str = OrderedDict(filter(
            lambda t: t[0] != "",
            zip(map(lambda s: s or "", channels), range(n_channels))))

        if len(channels) > n_channels:
            channels = channels[: n_channels]
        self._keys = OrderedDict(filter(
            lambda t: t[0] != "",
            zip_longest(range(n_channels), map(lambda s: s or "", channels), fillvalue="")))

        self._raw = np.empty((n_channels, n_samples), dtype="<f4", order="C")
        if data is not None:
            if n_channels == 1 and len(data.shape) == 1:
                self._raw[:] = data
            else:
                self._raw[:, :] = data

    # region Not Implemented Method

    def __contains__(self, item: np.ndarray) -> bool:
        raise TypeError

    def __delitem__(self, key: Union[int, str, Tuple[int, str]]) -> None:
        raise TypeError

    def pop(self, k: Union[int, str, Tuple[int, str]]) -> np.ndarray:
        raise TypeError

    def popitem(self) -> Tuple[Tuple[int, str], np.ndarray]:
        raise TypeError

    def clear(self) -> None:
        raise TypeError

    def update(self, __m, **kwargs) -> None:
        raise TypeError

    def setdefault(self, k: Union[int, str, Tuple[int, str]], default: np.ndarray = ...) -> np.ndarray:
        raise TypeError

    # endregion


class Monaural(Signal):

    def __init__(self, *args, n_samples: int, sampling_rate: int, data: Optional[np.ndarray] = None, **kwargs):
        super().__init__(n_channels=1, n_samples=n_samples, sampling_rate=sampling_rate, channels=["MONO"], data=data,
                         *args, **kwargs)

    # region ClassMethod

    @classmethod
    def create_wave(cls, wave_function: Callable[[int], float], *, seconds: float, sampling_rate: int) -> _monaural:
        n_samples: int = int(sampling_rate * seconds)
        data: np.ndarray = np.array([wave_function(n) for n in range(n_samples)],
                                    dtype="<f4", order="C")
        return Monaural(n_samples=n_samples, sampling_rate=sampling_rate, data=data)

    @classmethod
    def create_sine_wave(cls, frequency: float,
                         *, phase: float = 0, amplitude: float = 1,
                         seconds: float, sampling_rate: int) -> _monaural:
        return Monaural.create_wave(
            lambda n: amplitude * np.sin(2.0 * np.pi * frequency * n / sampling_rate + phase),
            seconds=seconds, sampling_rate=sampling_rate)

    @classmethod
    def create_triangle_wave(cls, frequency: float,
                             *, phase: float = 0, amplitude: float = 1,
                             seconds: float, sampling_rate: int) -> _monaural:
        return Monaural.create_wave(
            lambda n: amplitude * (-abs(4 * math.modf(frequency * n / sampling_rate + phase / (2 * np.pi))[0] - 2) + 1),
            seconds=seconds, sampling_rate=sampling_rate)

    @classmethod
    def create_sawtooth_wave(cls, frequency: float,
                             *, phase: float = 0, amplitude: float = 1,
                             seconds: float, sampling_rate: int) -> _monaural:
        return Monaural.create_wave(
            lambda n: amplitude * (2 * math.modf(frequency * n / sampling_rate + phase / (2 * np.pi))[0] - 1),
            seconds=seconds, sampling_rate=sampling_rate)

    @classmethod
    def create_square_wave(cls, frequency: float,
                           *, phase: float = 0, amplitude: float = 1,
                           seconds: float, sampling_rate: int) -> _monaural:
        return Monaural.create_wave(
            lambda n: amplitude * np.sign(np.sin(2.0 * np.pi * frequency * n / sampling_rate + phase)),
            seconds=seconds, sampling_rate=sampling_rate)

    @classmethod
    def create_silence(cls, *, seconds: float, sampling_rate: int) -> _monaural:
        return Monaural.create_wave(lambda n: 0, seconds=seconds, sampling_rate=sampling_rate)

    # endregion


class Stereo(Signal):

    def __init__(self, *args, n_samples: int, sampling_rate: int, data: Optional[np.ndarray] = None, **kwargs):
        super().__init__(n_channels=2, n_samples=n_samples, sampling_rate=sampling_rate, channels=["L", "R"], data=data,
                         *args, **kwargs)

    def get_mid_and_side_channel(self) -> Signal:
        signal: Signal = Signal(n_channels=self.number_of_channels,
                                n_samples=self.number_of_samples,
                                sampling_rate=self.sampling_rate,
                                channels=["L+R", "L-R"])
        l: np.ndarray = self["L"]
        r: np.ndarray = self["R"]
        signal["L+R"] = (l + r) / 2
        signal["L-R"] = (l - r) / 2
        return signal
