import io
from typing import Iterable, Optional

import numpy as np

from ..format import FormatTag, WaveFormat

from .waveiobase import WaveIOBase


class Writer(WaveIOBase, io.BufferedWriter):
    """ Wave形式のファイル内容を書き出す Writer です。 """

    def __init__(self, file: str = "", *args, **kwargs):
        super(Writer, self).__init__(file, *args, **kwargs)

    # region WaveIOBase Method

    def open(self, file: str = "") -> None:
        self._open("wb", file or self.file)

    def write(self, data: np.ndarray) -> int:

        self._raise_position_error()
        # data_bytes: bytes = b""

        if self.wave_format.format_code == FormatTag.pcm:
            raise NotImplementedError
        if self.wave_format.format_code == FormatTag.ieee_float:
            if data.dtype == "<f4":
                data_bytes: bytes = data.tobytes()
            else:
                data_bytes: bytes = np.array(data, dtype="<f4").tobytes()
        else:
            raise NotImplementedError

        pos: int = 0
        while pos < len(data_bytes):
            pos = pos + self._stream.write(data_bytes[pos:])

        return pos

    # endregion

    # region WaveIOBase Method (Not Implemented)

    def truncate(self, size: Optional[int] = ...) -> int:
        raise NotImplementedError

    def writelines(self, lines: Iterable[np.ndarray]) -> None:
        raise NotImplementedError

    # endregion

    # region BufferedWriter Method (Not Implemented)

    # def flush(self) -> None:
    #     super().flush()

    def peek(self, size: int = ...) -> bytes:
        raise NotImplementedError

    # endregion

    # region Writer Method

    def write_all(self, wave_format: WaveFormat, data: np.ndarray) -> None:
        self.write_header(wave_format, len(data))
        self.write(data)

    def write_header(self, wave_format: WaveFormat, data_size: int) -> None:

        if not self._stream.writable():
            raise OSError
        if self._stream.tell() != 0:
            if self._stream.seekable():
                self._stream.seek(0)
            else:
                raise OSError

        wf: WaveFormat = wave_format

        fmt_chunk_size: int = 16
        fact_chunk_size: int = 0
        if not wf.format_code.is_linear_pcm():
            fmt_chunk_size = 40 if wf.format_code == FormatTag.extensible else 18
            fact_chunk_size = len(wf.number_of_samples) * 4

        data_byte_size: int = data_size * wf.number_of_channels * wf.bytes_per_sample
        riff_chunk_size: int = 4 + (fmt_chunk_size + 8) + (fact_chunk_size + 8 if fact_chunk_size > 0 else 0) + (
                data_byte_size + 8)

        self._raw_position = 0
        self._write_chunk_header("RIFF", riff_chunk_size)
        self._write_str("WAVE")

        self._write_chunk_header("fmt ", fmt_chunk_size)

        self._write_int(wf.format_code.value, size=2)
        self._write_int(wf.number_of_channels, size=2)
        self._write_int(wf.sampling_rate, size=4)
        self._write_int(wf.data_rate, size=4)
        self._write_int(wf.data_block_size, size=2)
        self._write_int(wf.bits_per_sample, size=2)

        if not wf.format_code.is_linear_pcm():

            if wf.format_code == FormatTag.extensible:
                self._write_int(22, size=2)
                self._write_int(wf.number_of_valid_bits, size=2)
                self._write_int(wf.speaker_position_mask.value, size=4)
                self._stream.write(wf.sub_format)
                for i in range(len(wf.sub_format), 16):
                    self._write_str("\x00")
            else:
                self._write_int(0, size=2)

            self._write_chunk_header("fact", fact_chunk_size)
            for s in wf.number_of_samples:
                self._write_int(s, size=4)

        self._wave_format = wf
        self._has_wave_format = True

        self._write_chunk_header("data", data_byte_size)

        self._data_offset = self._raw_position
        self._data_size = data_byte_size

    # endregion

    # region Private Method

    def _write_chunk_header(self, chunk_id: str, chunk_size: int) -> None:
        self._write_str(chunk_id)
        self._write_int(chunk_size, size=4)

    def _write_int(self, value: int, *, size: int) -> None:
        self._stream.write(value.to_bytes(size, "little"))
        self._raw_position += size

    def _write_str(self, value: str) -> None:
        value_bytes: bytes = value.encode()
        self._stream.write(value_bytes)
        self._raw_position += len(value_bytes)

    # endregion
