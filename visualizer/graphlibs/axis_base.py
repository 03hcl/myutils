from abc import abstractmethod
from enum import auto, Enum, unique
from typing import Optional, Tuple, TypeVar

import numpy as np

from .graph_base import GraphBase
from .style import LineStyle

_direction = TypeVar("_direction", bound="Axis.Direction")


class Axis(GraphBase):

    # region Class

    @unique
    class Direction(Enum):
        Unknown = 0
        X = auto()
        Y = auto()
        Z = auto()

    @unique
    class Scale(Enum):
        Unknown = 0
        Linear = auto()
        Log = auto()
        SymmetricLog = auto()
        Logit = auto()
        Power = auto()

    # endregion

    def __init__(self, direction: Direction, *args, **kwargs):
        self._direction: _direction = direction
        super(Axis, self).__init__(*args, **kwargs)

    # region Property

    @property
    def direction(self) -> _direction:
        return self._direction

    # endregion

    @abstractmethod
    def label(self, value: Optional[str] = None, **kwargs) -> str:
        pass

    @abstractmethod
    def scale(self, value: Optional[Scale] = None, **kwargs) -> Scale:
        pass

    @abstractmethod
    def range(self, value: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[float, float]:
        pass

    @abstractmethod
    def scale_line(self, value: Optional[LineStyle] = None, **kwargs) -> LineStyle:
        pass

    @abstractmethod
    def minor_scale_line(self, value: Optional[LineStyle] = None, **kwargs) -> LineStyle:
        pass

    @abstractmethod
    def ticks(self, value: Optional[np.ndarray] = None, **kwargs) -> None:
        pass

    @abstractmethod
    def minor_ticks(self, value: Optional[np.ndarray] = None, **kwargs) -> None:
        pass

    @abstractmethod
    def tick_labels(self, value: Optional[np.ndarray] = None, **kwargs) -> None:
        pass

    @abstractmethod
    def minor_tick_labels(self, value: Optional[np.ndarray] = None, **kwargs) -> None:
        pass


AxisBase = Axis
