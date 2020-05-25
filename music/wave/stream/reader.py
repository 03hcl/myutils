import io
import os
from typing import Optional, List, Tuple

import numpy as np

from ..format import ChannelMask, FormatTag, WaveFormat

from .waveiobase import WaveIOBase
from .exceptions import IncorrectWaveFormatError, NotFoundDataChunkError


class Reader(WaveIOBase, io.BufferedReader):
    """ Wave形式のファイル内容を読み込む Reader です。 """

    def __init__(self, file: str = "", *args, **kwargs):
        super(Reader, self).__init__(file, *args, **kwargs)

    # region WaveIOBase Method

    def open(self, file: str = "") -> None:
        self._open("rb", file or self.file)

    def read(self, size: Optional[int] = None) -> Tuple[np.ndarray, bytes]:

        size = self._get_size(size)
        if self.wave_format.format_code == FormatTag.ieee_float or self.wave_format.format_code == FormatTag.pcm:
            data_array = np.empty([size, self.wave_format.number_of_channels], dtype="<f4")
        else:
            raise NotImplementedError

        pos: int = 0
        end_buffer: bytes = b""

        while pos < size and self._raw_position < self.data_offset + self.data_size:
            data, end_buffer = self.read1(size - pos, end_buffer)
            data_array[pos: pos + len(data)] = data
            pos += len(data)

        return data_array, end_buffer

    def read1(self, size: Optional[int] = None, head_buffer: bytes = b"") -> Tuple[np.ndarray, bytes]:

        sample_size: int = self.wave_format.number_of_channels * self.wave_format.bytes_per_sample
        size = self._get_size(size)
        self._raise_position_error()

        data: bytes = b"".join([head_buffer, self._stream.read(size * sample_size)])

        if self._raw_position + len(data) > self.data_offset + self.data_size:
            data = data[: self.data_offset + self.data_size - self._raw_position]
        self._raw_position += len(data)

        end_buffer: bytes = b""
        if len(data) % sample_size != 0:
            end_buffer = data[-(len(data) % sample_size):]
            data = data[: -(len(data) % sample_size)]

        # noinspection PyUnusedLocal
        data_array: np.ndarray

        if self.wave_format.format_code == FormatTag.ieee_float:
            data_array = np.frombuffer(data, dtype="<f4")
        elif self.wave_format.format_code == FormatTag.pcm:
            if self.wave_format.bits_per_sample == 8:
                data_array = np.frombuffer(data, dtype="u1")
                data_array = (data_array.astype(dtype="<f4") - 128) / 128
            elif self.wave_format.bits_per_sample == 16:
                data_array = np.frombuffer(data, dtype="<i2")
                data_array = data_array.astype(dtype="<f4") / 32768
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return np.reshape(data_array, [size, self.wave_format.number_of_channels]), end_buffer

    # endregion

    # region WaveIOBase Method (Not Implemented)

    def readinto(self, b: np.ndarray) -> int:
        raise NotImplementedError

    def readinto1(self, b: np.ndarray) -> int:
        raise NotImplementedError

    def readline(self, size: int = ...) -> np.ndarray:
        raise NotImplementedError

    def readlines(self, hint: int = ...) -> List[np.ndarray]:
        raise NotImplementedError

    # endregion

    # region BufferedReader Method (Not Implemented)

    def peek(self, size: int = ...) -> np.ndarray:
        raise NotImplementedError

    # endregion

    # region Reader Method

    def read_header(self) -> WaveFormat:
        """ data チャンクのデータが始まる手前までのチャンクを読み込み WaveFormat を読み込んでいればそれを返します。"""

        if not self._stream.readable():
            raise OSError
        if self._stream.tell() != 0:
            if self._stream.seekable():
                self._stream.seek(0)
            else:
                raise OSError

        self._raw_position = 0
        riff_id, riff_size = self._read_chunk_header()

        if riff_id != "RIFF":
            raise IncorrectWaveFormatError(message="RIFF フォーマットとして認識されませんでした。")

        if self._read_str(4) != "WAVE":
            raise IncorrectWaveFormatError(message="WAVE 形式として認識されませんでした。")

        while self._raw_position < riff_size + 8:

            chunk_id, chunk_size = self._read_chunk_header()
            start_position = self._raw_position

            if chunk_id == "fmt ":

                wf = WaveFormat()
                wf.format_code = FormatTag(self._read_int(2))

                if chunk_size < (16 if wf.format_code.is_linear_pcm() else 18):
                    raise IncorrectWaveFormatError("fmt チャンクのサイズが想定外です。")

                wf.number_of_channels = self._read_int(2)
                wf.sampling_rate = self._read_int(4)
                wf.data_rate = self._read_int(4)
                wf.data_block_size = self._read_int(2)
                wf.bits_per_sample = self._read_int(2)

                if wf.sampling_rate * wf.data_block_size != wf.data_rate:
                    raise IncorrectWaveFormatError("チャンネル数、サンプリングレートとサンプル長が整合しません？")

                if wf.format_code.is_linear_pcm():
                    self._skip_stream(chunk_size - 16)
                else:
                    extension_size = self._read_int(2)
                    if chunk_size < extension_size + 18:
                        raise IncorrectWaveFormatError("fmt チャンクのサイズとチャンク内の拡張部分のサイズが整合しません。")
                    if wf.format_code == FormatTag.extensible:
                        if extension_size < 22:
                            raise IncorrectWaveFormatError("extensible フォーマットにおける拡張部分のサイズが整合しません。")
                        wf.number_of_valid_bits = self._read_int(2)
                        wf.speaker_position_mask = ChannelMask(self._read_int(4))
                        wf.sub_format = self._stream.read(16)
                        self._raw_position += 16
                        self._skip_stream(extension_size - 22)
                    else:
                        self._skip_stream(extension_size)

                self._wave_format = wf
                self._has_wave_format = True

            elif chunk_id == "data":

                self._data_offset = self._raw_position
                self._data_size = chunk_size
                return self._wave_format

            elif chunk_id == "fact":
                # Not Implemented
                pass

            else:

                self._skip_stream(chunk_size)

            self._skip_stream(self._raw_position - start_position - chunk_size)

        raise NotFoundDataChunkError

    def read_all(self) -> Tuple[WaveFormat, np.ndarray]:
        wf = self.read_header()
        (data, _) = self.read()
        return wf, data

    # endregion

    # region Private Method

    def _get_size(self, size: Optional[int]) -> int:
        if size and size >= 0:
            return size
        if not self.wave_format:
            raise IncorrectWaveFormatError("WaveFormat が読み込まれていないため、データの最大サイズがわかりません。")
        sample_size: int = self.wave_format.number_of_channels * self.wave_format.bytes_per_sample
        return self.data_size // sample_size

    def _read_chunk_header(self) -> (str, int):
        chunk_id = self._read_str(4)
        chunk_size = self._read_int(4)
        return chunk_id, chunk_size

    def _read_int(self, size: int) -> int:
        value = int.from_bytes(self._stream.read(size), "little")
        self._raw_position += size
        return value

    def _read_str(self, size: int) -> str:
        value = self._stream.read(size).decode()
        self._raw_position += size
        return value

    def _skip_stream(self, size: int) -> None:
        if size <= 0:
            return
        if self._stream.seekable():
            self._stream.seek(size, os.SEEK_CUR)
        elif self._stream.readable():
            self._stream.read(size)
        else:
            raise OSError
        self._raw_position += size

    # endregion
