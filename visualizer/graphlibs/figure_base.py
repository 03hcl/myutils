from abc import abstractmethod
from types import TracebackType
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

from .axes_base import Axes
from .graph_base import GraphBase

_figure = TypeVar("_figure", bound="Figure")


class Figure(GraphBase):

    def __init__(self, *args: Any, **kwargs: Any):
        self._init_args: Tuple[Any, ...] = args
        self._init_kwargs: Dict[str, Any] = kwargs
        super(Figure, self).__init__(*args, **kwargs)

    # region Special Method

    @abstractmethod
    def __enter__(self) -> _figure:
        pass

    @abstractmethod
    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        pass

    # endregion

    @abstractmethod
    def open(self, grid: Tuple[int, int] = (1, 1), *args, **kwargs) -> object:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def window_title(self, value: Optional[str]) -> Optional[str]:
        pass

    @abstractmethod
    def axes(self, location: Tuple[int, int] = (0, 0), span: Tuple[int, int] = (1, 1)) -> Axes:
        pass

    @abstractmethod
    def save_as_png(self, directory_path: str, file_name: str) -> None:
        pass

    @abstractmethod
    def show(self) -> None:
        pass


FigureBase = Figure
