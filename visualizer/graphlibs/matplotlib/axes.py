from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from matplotlib.axes import Axes as AxesRaw
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure as FigureRaw
from matplotlib.legend import Legend

from .. import AxesBase, CollidePlotLabelError, LineStyle, MarkerStyle

from .axisx import AxisX
from .axisy import AxisY
from .colorbar import ColorBar
from .style import line_style_to_str, marker_style_to_str

Orientation = AxesBase.Orientation


class Axes(AxesBase):

    def __init__(self, axes: AxesRaw, figure: FigureRaw, *args, **kwargs):

        self._raw: AxesRaw = axes
        self._figure: FigureRaw = figure

        self._title: str = self._raw.get_title()
        self._axis_x: AxisX = AxisX(self._raw.get_xaxis(), self._raw)
        self._axis_y: AxisY = AxisY(self._raw.get_yaxis(), self._raw)

        self._color_bar_orientation: Optional[Orientation] = None
        self._color_bar: Optional[ColorBar] = None
        self._legend: Optional[Legend] = None
        self._legend_visible: bool = False
        self._legend_dict: dict = {}

        self._mappable: Optional[QuadMesh] = None

        super(Axes, self).__init__(*args, **kwargs)

    # region AxesBase Method

    def axis_x(self, **kwargs) -> AxisX:
        return self._axis_x

    def axis_y(self, **kwargs) -> AxisY:
        return self._axis_y

    def color_bar(self, visible: Optional[bool] = None, orientation: Optional[Orientation] = None,
                  **kwargs) -> Optional[ColorBar]:
        if self._mappable is None:
            if self._color_bar is not None:
                self._delete_color_bar()
        else:
            if visible:
                if self._color_bar is not None and orientation != self._color_bar_orientation:
                    self._delete_color_bar()
                if self._color_bar is None:
                    self._create_color_bar(orientation, **kwargs)
            if not visible and self._color_bar is not None:
                self._delete_color_bar()
        return self._color_bar

    def legend(self, visible: Optional[bool] = None, **kwargs) -> None:
        if visible is not None:
            if visible:
                if self._legend is None:
                    self._legend = self._raw.legend(**kwargs)
                    self._legend_visible = True
                    self._legend_dict = kwargs
                elif not self._legend_visible:
                    self._legend_visible = True
                    self._legend.set_visible(self._legend_visible)
            elif self._legend_visible:
                self._legend_visible = False
                self._legend.set_visible(self._legend_visible)

    def plot(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
             marker_style: MarkerStyle = MarkerStyle.Dot, **kwargs) -> None:
        x = Axes._create_x(y) if x is None else x
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        self._raw.plot(x, y, linewidth=0, marker=marker_style_to_str(marker_style), **kwargs)
        self._recreate_legend()

    def line(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
             line_style: LineStyle = LineStyle.Solid, **kwargs) -> None:
        x = Axes._create_x(y) if x is None else x
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        self._raw.plot(x, y, linestyle=line_style_to_str(line_style), **kwargs)
        self._recreate_legend()

    def histogram(self, data: np.ndarray, label: Optional[str] = None,
                  bins: int = 10, data_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  line_style: LineStyle = LineStyle.Solid, orientation: Orientation = Orientation.Vertical,
                  **kwargs) -> None:
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        self._raw.hist(x=data, bins=bins, range=Axes._data_range(data_range, data),
                       linestyle=line_style_to_str(line_style), orientation=str(orientation), **kwargs)
        self._recreate_legend()

    def clustered_histogram(
            self, data: Union[np.ndarray, Iterable[np.ndarray]], label: Optional[Iterable[str]] = None,
            bins: int = 10, data_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
            line_style: LineStyle = LineStyle.Solid, orientation: Orientation = Orientation.Vertical,
            **kwargs) -> None:
        data: List[np.ndarray] = \
            [data[i] for i in range(0, data.shape[0])] if issubclass(type(data), np.ndarray) else list(data)
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        self._raw.hist(x=data, bins=bins, range=Axes._data_range(data_range, data),
                       linestyle=line_style_to_str(line_style), orientation=str(orientation), **kwargs)
        self._recreate_legend()

    def bar(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
            width: float = 1, orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        x = Axes._create_x(y) if x is None else x
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        if orientation == Orientation.Vertical:
            self._raw.bar(x=x, height=y, width=width, **kwargs)
        elif orientation == Orientation.Horizontal:
            # noinspection PyTypeChecker
            self._raw.barh(y=x, width=y, height=width, **kwargs)
        else:
            raise NotImplementedError
        self._recreate_legend()

    def clustered_bar(self, x: Optional[np.ndarray], y: Union[np.ndarray, Iterable[np.ndarray]],
                      label: Optional[Iterable[str]] = None, width: float = 0.9,
                      orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        x = Axes._create_x(y) if x is None else x
        data: List[np.ndarray] = [y[i] for i in range(0, y.shape[0])] if issubclass(type(y), np.ndarray) else list(y)
        each_width: float = width / len(data)
        for i, (y_i, l_i) in enumerate(zip(data, label)):
            kwargs_i = Axes._add_label_to_kwargs(l_i, kwargs)
            if orientation == Orientation.Vertical:
                self._raw.bar(x=x + (i - (len(data) - 1) / 2) * each_width, height=y_i, width=each_width, **kwargs_i)
            elif orientation == Orientation.Horizontal:
                # noinspection PyTypeChecker
                self._raw.barh(y=x + (i - (len(data) - 1) / 2) * each_width, width=y_i, height=each_width, **kwargs_i)
            else:
                raise NotImplementedError

    def stacked_bar(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
                    width: float = 1, orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        x = Axes._create_x(y) if x is None else x
        accumulated: np.ndarray = np.zeros(y.shape[1], dtype=float)
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        for i in range(0, y.shape[0]):
            if orientation == Orientation.Vertical:
                self._raw.bar(x=x, height=y[i, :], width=width, bottom=accumulated, **kwargs)
            elif orientation == Orientation.Horizontal:
                # noinspection PyTypeChecker
                self._raw.barh(y=x, width=y[i, :], height=width, bottom=accumulated, **kwargs)
            else:
                raise NotImplementedError
            accumulated += y[i, :]
        self._recreate_legend()

    def scatter(self, x: np.ndarray, y: np.ndarray, label: Optional[str] = None, **kwargs) -> None:
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        self._raw.scatter(x=x, y=y, marker=".", **kwargs)
        self._recreate_legend()

    def heatmap(self, data: np.ndarray, label: Optional[str] = None,
                data_range: Tuple[Optional[float], Optional[float]] = (None, None),
                **kwargs) -> None:
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        self._mappable = self._raw.imshow(X=data, vmin=data_range[0], vmax=data_range[1], **kwargs)
        self._recreate_legend()

    def title(self, value: Optional[str] = None, font_size: Optional[int] = None, **kwargs) -> str:
        if value is not None:
            self._raw.set_title(value, fontsize=font_size, **kwargs)
            self._title = self._raw.get_title()
        return self._title

    # endregion

    # region GraphBase Method

    def clear(self) -> None:
        self._raw.cla()

    # endregion

    # region Private Method

    @staticmethod
    def _add_label_to_kwargs(label: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if label is None:
            return kwargs
        if "label" in kwargs:
            raise CollidePlotLabelError
        return {**kwargs, "label": label}

    def _recreate_legend(self, **kwargs) -> None:
        if not self._legend:
            return
        if kwargs:
            self._legend_dict = kwargs
        self._legend.remove()
        self._legend = self._raw.legend(**self._legend_dict)
        self._legend.set_visible(self._legend_visible)

    def _create_color_bar(self, orientation: Optional[Orientation] = None, **kwargs) -> None:
        from matplotlib import pyplot as plt
        self._color_bar_orientation = orientation or Orientation.Vertical
        self._color_bar = ColorBar(plt.colorbar(self._mappable, ax=self._raw,
                                                orientation=str(self._color_bar_orientation), **kwargs),
                                   self._figure)

    def _delete_color_bar(self) -> None:
        self._color_bar.clear()
        self._color_bar_orientation = None
        self._color_bar = None

    # endregion
