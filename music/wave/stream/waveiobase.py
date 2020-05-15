import abc
import io
import os
from types import TracebackType
from typing import Iterable, Callable, List, Optional, Type, TypeVar

import numpy as np

from ..format import WaveFormat
from .exceptions import UnexpectedStreamPositionError, UnreadDataOffsetError

_waveio_base = TypeVar("_waveio_base", bound="WaveIOBase")


class WaveIOBase(io.BufferedIOBase):

    # region Property

    @property
    def file(self) -> str:
        return self._file

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def has_wave_format(self) -> bool:
        return self._has_wave_format

    @property
    def wave_format(self) -> Optional[WaveFormat]:
        return self._wave_format

    @property
    def data_offset(self) -> Optional[int]:
        return self._data_offset

    @property
    def data_size(self) -> Optional[int]:
        return self._data_size

    # endregion

    def __init__(self, file: str = "", *args, **kwargs):
        self._file: str = file
        self._closed: bool = True
        self._has_wave_format: bool = False
        self._wave_format: Optional[WaveFormat] = None
        self._data_offset: Optional[int] = None
        self._data_size: Optional[int] = None
        self._stream: Optional[io.BytesIO] = None
        self._raw_position: Optional[int] = None
        # super(WaveIOBase, self).__init__(*args, **kwargs)

    # region Special Method

    # def __del__(self) -> None:
    #     self.close()

    def __enter__(self) -> _waveio_base:
        self.open()
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> bool:
        self.close()
        return False

    def __iter__(self):
        # return iter(self._stream)
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration()
        else:
            return next(self._stream)

    # endregion

    # region BufferedIOBase Method

    def close(self) -> None:
        """ストリームが開いていれば閉じます。"""
        if not self.closed:
            self._stream.close()
            self._initialize_property()

    def detach(self) -> io.RawIOBase:
        return self._stream.detach()

    def fileno(self) -> int:
        return self._stream.fileno()

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def readable(self) -> bool:
        return self._is_raised_position_error_wrapper(self._stream.readable)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            if offset > self.data_size or offset < 0:
                raise UnexpectedStreamPositionError("data チャンク外へはシークできません。")
            self._stream.seek(self.data_offset + offset, os.SEEK_SET)
            self._raw_position = self.data_offset + offset
        elif whence == os.SEEK_CUR:
            # raw_position = self._tell_raw_position()
            if offset > self.data_size - self._raw_position or offset < - self._raw_position:
                raise UnexpectedStreamPositionError("data チャンク外へはシークできません。")
            self._stream.seek(offset, os.SEEK_CUR)
            self._raw_position += offset
        elif whence == os.SEEK_END:
            if offset > 0 or offset < - self._data_size:
                raise UnexpectedStreamPositionError("data チャンク外へはシークできません。")
            self._stream.seek(self.data_offset + self.data_size + offset, os.SEEK_SET)
            self._raw_position = self.data_offset + self.data_size + offset
        else:
            raise NotImplementedError
        return self._raw_position - self.data_offset

    def seekable(self) -> bool:
        return self._is_raised_position_error_wrapper(self._stream.seekable)

    def tell(self) -> int:
        self._raise_position_error()
        return self._raw_position - self.data_offset

    def writable(self) -> bool:
        return self._is_raised_position_error_wrapper(self._stream.writable)

    # endregion

    # region BufferedIOBase Method (Not Implemented)

    def read(self, size: Optional[int] = ...) -> np.ndarray:
        raise TypeError

    def read1(self, size: int = ...) -> np.ndarray:
        raise TypeError

    def readinto(self, b: np.ndarray) -> int:
        raise TypeError

    def readinto1(self, b: np.ndarray) -> int:
        raise TypeError

    def readline(self, size: int = ...) -> np.ndarray:
        raise TypeError

    def readlines(self, hint: int = ...) -> List[np.ndarray]:
        raise TypeError

    def truncate(self, size: Optional[int] = ...) -> int:
        raise TypeError

    def write(self, b: np.ndarray) -> int:
        raise TypeError

    def writelines(self, lines: Iterable[np.ndarray]) -> None:
        raise TypeError

    # endregion

    # region WaveIOBase Method

    @classmethod
    def create(cls: type(_waveio_base), file: str) -> _waveio_base:
        """stream を作成して開きます。"""
        instance = cls()
        instance.open(file)
        return instance

    @abc.abstractmethod
    def open(self, file: str = ...) -> None:
        pass

    # endregion

    # region Private Method

    def _initialize_property(self):
        self._file = ""
        self._closed = True
        self._has_wave_format = False
        self._wave_format = None
        self._data_offset = None
        self._data_size = None
        self._stream = None
        self._raw_position = None

    def _is_raised_position_error_wrapper(self, func: Callable[[], bool]) -> bool:
        try:
            self._raise_position_error()
        except (UnexpectedStreamPositionError, UnreadDataOffsetError):
            return False
        return func()

    def _open(self, mode: str, file: str = None) -> None:
        if not self.closed:
            self.close()
        if file:
            self._file = file
        self._stream = io.open(self._file, mode)
        self._closed = False

    def _raise_position_error(self) -> None:
        if self._raw_position is None:
            raise UnexpectedStreamPositionError("ストリームの読み込みまたは書き込みを開始していません。")
        if self.data_size is None or self.data_offset is None:
            raise UnreadDataOffsetError("data チャンクのサイズまたはオフセットが不明です。")
        if self._raw_position < self.data_offset or self._raw_position > self.data_offset + self.data_size:
            raise UnexpectedStreamPositionError("現在のストリーム位置が data チャンク内にありません。")

    # endregion
