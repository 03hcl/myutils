from abc import abstractmethod
from enum import auto, Enum, unique
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from .axis_base import Axis
from .color_bar_base import ColorBar
from .graph_base import GraphBase
from .style import LineStyle, MarkerStyle


class Axes(GraphBase):

    @unique
    class Orientation(Enum):
        Unknown = 0
        Vertical = auto()
        Horizontal = auto()

        def __str__(self):
            if self == Axes.Orientation.Vertical:
                return "vertical"
            if self == Axes.Orientation.Horizontal:
                return "horizontal"
            raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super(Axes, self).__init__(*args, **kwargs)

    @abstractmethod
    def axis_x(self, **kwargs) -> Axis:
        pass

    @abstractmethod
    def axis_y(self, **kwargs) -> Axis:
        pass

    @abstractmethod
    def color_bar(self, visible: Optional[bool] = None, orientation: Optional[Orientation] = None,
                  **kwargs) -> ColorBar:
        pass

    @abstractmethod
    def legend(self, visible: Optional[bool] = None, **kwargs) -> None:
        pass

    @abstractmethod
    def plot(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
             marker_style: MarkerStyle = MarkerStyle.Dot, **kwargs) -> None:
        pass

    @abstractmethod
    def line(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
             line_style: LineStyle = LineStyle.Solid, **kwargs) -> None:
        pass

    @abstractmethod
    def histogram(self, data: np.ndarray, label: Optional[str] = None,
                  bins: int = 10, data_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  line_style: LineStyle = LineStyle.Solid, orientation: Orientation = Orientation.Vertical,
                  **kwargs) -> None:
        pass

    @abstractmethod
    def clustered_histogram(
            self, data: Union[np.ndarray, Iterable[np.ndarray]], label: Optional[Iterable[str]] = None,
            bins: int = 10, data_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
            line_style: LineStyle = LineStyle.Solid, orientation: Orientation = Orientation.Vertical,
            **kwargs) -> None:
        pass

    @abstractmethod
    def bar(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
            width: float = 1, orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        pass

    @abstractmethod
    def clustered_bar(self, x: Optional[np.ndarray], y: Union[np.ndarray, Iterable[np.ndarray]],
                      label: Optional[Iterable[str]] = None, width: float = 0.9,
                      orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        pass

    @abstractmethod
    def stacked_bar(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
                    width: float = 1, orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        pass

    @abstractmethod
    def scatter(self, x: np.ndarray, y: np.ndarray, label: Optional[str] = None, **kwargs) -> None:
        pass

    @abstractmethod
    def heatmap(self, data: np.ndarray, label: Optional[str] = None,
                data_range: Tuple[Optional[float], Optional[float]] = (None, None),
                **kwargs) -> None:
        pass

    @abstractmethod
    def title(self, value: Optional[str] = None, font_size: Optional[int] = None, **kwargs) -> str:
        pass

    def grid(self, line_style: LineStyle = LineStyle.Solid) -> None:
        self.axis_x().scale_line(line_style)
        self.axis_y().scale_line(line_style)

    def minor_grid(self, line_style: LineStyle = LineStyle.Solid) -> None:
        self.grid()
        self.axis_x().minor_scale_line(line_style)
        self.axis_y().minor_scale_line(line_style)

    @staticmethod
    def _data_range(value: Optional[Tuple[Optional[float], Optional[float]]],
                    data: Union[np.ndarray, Iterable[np.ndarray]]) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        data_min: float = (min(data) if issubclass(type(data), np.ndarray) else min(min(d) for d in data)) \
            if value[0] is None else value[0]
        data_max: float = (max(data) if issubclass(type(data), np.ndarray) else max(max(d) for d in data)) \
            if value[1] is None else value[1]
        return data_min, data_max

    @staticmethod
    def _create_x(y: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        length: int = len(y) if issubclass(type(y), np.ndarray) else max(len(y_i) for y_i in y)
        return np.arange(0, length)


AxesBase = Axes
