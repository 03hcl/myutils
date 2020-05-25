from typing import Iterable, Optional, Tuple, Union

from cv2 import VideoWriter, VideoWriter_fourcc

import numpy as np

from .. import WriterBase


class Writer(WriterBase):

    def __init__(self, file: str,
                 fourcc: Union[int, str, Iterable[str]], fps: float, frame_size: Tuple[int, int], is_color: bool = True,
                 *args, **kwargs):
        super(Writer, self).__init__(file, args, kwargs)
        self._fourcc: int = _fourcc(fourcc)
        self._fps: float = fps
        self._frame_size: Tuple[int, int] = frame_size
        self._is_color: bool = is_color
        self._stream: Optional[VideoWriter] = None

    def write_frame(self, frame: np.ndarray) -> None:
        self._stream.write(frame)
        self._position += 1

    def open(self, file: str = "", *args, **kwargs) -> None:
        if not self.closed:
            self.close()
        if file:
            self._file = file
        self._stream = VideoWriter(self._file, self._fourcc, self._fps, self._frame_size, self._is_color)
        self._closed = False

    def close(self) -> None:
        if not self.closed:
            self._stream.release()
            self._initialize_property()


def _fourcc(value: Union[int, str, Iterable[str]]) -> int:
    if issubclass(type(value), int):
        return value
    if issubclass(type(value), str):
        return VideoWriter_fourcc(*value)
    if issubclass(type(value), tuple) or issubclass(type(value), list):
        return VideoWriter_fourcc(*value)
    raise NotImplementedError
