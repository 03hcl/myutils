import abc
from types import TracebackType
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Type, TypeVar, Union

import numpy as np

_writer = TypeVar("_writer", bound="Writer")


class Writer:

    # region Property

    @property
    def file(self) -> str:
        return self._file

    @property
    def closed(self) -> bool:
        return self._closed

    # endregion

    def __init__(self, file: str, *args: Any, **kwargs: Any):
        self._file: str = file
        self._closed: bool = True
        self._position: int = 0
        self._init_args: Tuple[Any, ...] = args
        self._init_kwargs: Dict[str, Any] = kwargs

    # region Abstract Method

    @abc.abstractmethod
    def open(self, file: str = ..., *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def write_frame(self, frame: np.ndarray) -> None:
        pass

    # endregion

    def write_frames(self, frames: Union[np.ndarray, Iterable[np.ndarray]]) -> None:

        # noinspection PyUnusedLocal
        iterator: Iterator[np.ndarray]

        if issubclass(type(frames), np.ndarray):
            iterator = iter(frames[i] for i in range(frames.shape[0]))
        else:
            iterator = iter(frames)

        for frame in iterator:
            self.write_frame(frame)

    # region Special Method

    def __enter__(self) -> _writer:
        self.open(*self._init_args, **self._init_kwargs)
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> bool:
        self.close()
        return False

    # endregion

    # region Private Method

    def _initialize_property(self):
        self._file = ""
        self._closed = True
        self._position = 0

    # endregion


WriterBase = Writer
