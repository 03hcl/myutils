from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from pyqtgraph import BarGraphItem, GraphicsLayout, HistogramLUTItem, ImageItem, LegendItem, \
    PlotDataItem, PlotItem, ScatterPlotItem

from .. import AxesBase, CollidePlotLabelError, LineStyle, MarkerStyle, PropertyBase

from .axisx import AxisX
from .axisy import AxisY
from .colorbar import ColorBar
from .style import marker_style_to_dict

Orientation = AxesBase.Orientation


class Axes(AxesBase):

    # region Class

    class Title(PropertyBase[str]):

        def __init__(self, axes: PlotItem, *args, **kwargs):
            self._axes: PlotItem = axes
            super(Axes.Title, self).__init__(*args, **kwargs)

        def _get(self, **kwargs) -> str:
            return self._axes.titleLabel.text

        def _set(self, value: str, **kwargs) -> None:
            self._axes.setTitle(value, **kwargs)

    # endregion

    def __init__(self, view: GraphicsLayout, figure: GraphicsLayout, *args, **kwargs):

        self._view: GraphicsLayout = view
        self._figure: GraphicsLayout = figure

        self._raw: PlotItem = self._view.addPlot()
        # if visible_axis:
        #     self._raw: PlotItem = self._view.addPlot()
        # else:
        #     self._raw: ViewBox = self._view.addViewBox()

        self._axis_x: AxisX = AxisX(self._raw)
        self._axis_y: AxisY = AxisY(self._raw)
        self._title: Axes.Title = Axes.Title(self._raw)

        self._image: Optional[ImageItem] = None
        # self._images: List[PlotItem] = []
        self._data_range: Optional[Tuple[Optional[float], Optional[float]]] = None
        self._color_bar: Optional[ColorBar] = None
        self._legend: Optional[LegendItem] = None

        super(Axes, self).__init__(*args, **kwargs)

    # region Axes Method

    def remove_from_figure(self) -> None:
        self._figure.removeItem(self._raw)

    # endregion

    # region AxesBase Method

    def axis_x(self, **kwargs) -> AxisX:
        return self._axis_x

    def axis_y(self, **kwargs) -> AxisY:
        return self._axis_y

    def color_bar(self, visible: Optional[bool] = None, orientation: Optional[Orientation] = None,
                  **kwargs) -> ColorBar:
        # if self._images is None:
        if self._image is None:
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
                self._legend = self._add_legend(**kwargs)
            elif self._legend is not None:
                self._raw.removeItem(self._legend)

    def plot(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
             marker_style: MarkerStyle = MarkerStyle.Dot, **kwargs) -> None:
        x = Axes._create_x(y) if x is None else x
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        # self._raw.plot(x=x, y=y, pen=None, **marker_style_to_dict(marker_style), **kwargs)
        self._raw.addItem(PlotDataItem(x=x, y=y, pen=None, **marker_style_to_dict(marker_style), **kwargs))

    def line(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
             line_style: LineStyle = LineStyle.Solid, **kwargs) -> None:
        x = Axes._create_x(y) if x is None else x
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        self._raw.addItem(PlotDataItem(x=x, y=y, **kwargs))

    def histogram(self, data: np.ndarray, label: Optional[str] = None,
                  bins: int = 10, data_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  line_style: LineStyle = LineStyle.Solid, orientation: Orientation = Orientation.Vertical,
                  **kwargs) -> None:
        hist, bin_list = np.histogram(data, bins=bins, range=Axes._data_range(data_range, data))
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        if orientation == Orientation.Vertical:
            self._raw.addItem(PlotDataItem(x=bin_list, y=hist, stepMode=True, **kwargs))
        else:
            raise NotImplementedError

    def clustered_histogram(
            self, data: Union[np.ndarray, Iterable[np.ndarray]], label: Optional[Iterable[str]] = None,
            bins: int = 10, data_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
            line_style: LineStyle = LineStyle.Solid, orientation: Orientation = Orientation.Vertical,
            **kwargs) -> None:
        raise NotImplementedError

    def bar(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
            width: float = 1, orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        x = Axes._create_x(y) if x is None else x
        if orientation == Orientation.Vertical:
            self._raw.addItem(BarGraphItem(x=x, height=y, width=width, **kwargs))
        else:
            raise NotImplementedError

    def clustered_bar(self, x: Optional[np.ndarray], y: Union[np.ndarray, Iterable[np.ndarray]],
                      label: Optional[Iterable[str]] = None, width: float = 0.9,
                      orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        raise NotImplementedError

    def stacked_bar(self, x: Optional[np.ndarray], y: np.ndarray, label: Optional[str] = None,
                    width: float = 1, orientation: Orientation = Orientation.Vertical, **kwargs) -> None:
        x = Axes._create_x(y) if x is None else x
        accumulated: np.ndarray = np.zeros(y.shape[1], dtype=float)
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        for i in range(0, y.shape[0]):
            if orientation == Orientation.Vertical:
                self._raw.addItem(BarGraphItem(x=x, y=accumulated + y[i, :] / 2, height=y[i, :], width=width, **kwargs))
            else:
                raise NotImplementedError
            accumulated += y[i, :]

    def scatter(self, x: np.ndarray, y: np.ndarray, label: Optional[str] = None,
                symbol: Optional[list] = None, **kwargs) -> None:
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        item = ScatterPlotItem(**kwargs)
        item.setData(x=x, y=y)
        self._raw.addItem(item)

    def heatmap(self, data: np.ndarray, label: Optional[str] = None,
                data_range: Tuple[Optional[float], Optional[float]] = (None, None),
                **kwargs) -> None:
        kwargs = Axes._add_label_to_kwargs(label, kwargs)
        self._image = ImageItem(data.T, **kwargs)
        self._raw.addItem(self._image)
        # item = ImageItem(data.T, **kwargs)
        # self._images.append(item)
        # self._raw.addItem(item)
        # if data_range[0] is not None and data_range[1] is not None:
        self._data_range = data_range

    def title(self, value: Optional[str] = None, font_size: Optional[int] = None, **kwargs) -> str:
        return self._title(value, size=str(font_size) + "pt", **kwargs)

    # endregion

    # region GraphBase Method

    def clear(self) -> None:
        self._view.removeItem(self._raw)
        self._image = None

    # endregion

    # region Private Method

    @staticmethod
    def _add_label_to_kwargs(label: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if label is None:
            return kwargs
        if "name" in kwargs:
            raise CollidePlotLabelError
        return {**kwargs, "name": label}

    def _add_legend(self, **kwargs) -> LegendItem:
        legend: LegendItem = self._raw.addLegend(**kwargs)
        for item in self._raw.listDataItems():
            if "name" in item.opts:
                legend.addItem(item, name=item.opts["name"])
        return legend

    def _create_color_bar(self, orientation: Optional[Orientation] = None, **kwargs) -> None:

        self._color_bar_orientation = orientation or Orientation.Vertical
        if self._color_bar_orientation == Orientation.Vertical:
            color_bar: HistogramLUTItem = HistogramLUTItem(self._image, **kwargs)
            # color_bar: HistogramLUTItem = HistogramLUTItem(**kwargs)
            # for item in self._images:
            #     self._color_bar.setImageItem(item)
        else:
            raise NotImplementedError

        self._view.addItem(color_bar, row=0, col=1)
        # self._raw.addItem(self._color_bar)

        self._color_bar = ColorBar(color_bar, self._figure, self._image)

        if self._data_range is not None:
            self._color_bar.range(self._data_range)
            # self._color_bar.range(self._data_range, padding=self._raw.getViewBox().suggestPadding(color_bar.axis))

    def _delete_color_bar(self) -> None:
        self._color_bar.clear()
        self._color_bar_orientation = None
        self._raw.removeItem(self._color_bar)
        self._color_bar = None

    # endregion


class FixedLegendItem(LegendItem):
    def __init__(self,  **kwargs):
        super(FixedLegendItem, self).__init__(**kwargs)
        self.getViewBox()
