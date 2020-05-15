from abc import ABC
from typing import Any, List, Optional, Tuple, Type

import numpy as np
from pyqtgraph import AxisItem, PlotItem

from .. import AxisBase, LineStyle, PropertyBase

Direction = AxisBase.Direction
Scale = AxisBase.Scale


class Axis(AxisBase):

    # region Class

    class _Property(PropertyBase, ABC):

        def __init__(self, axis: AxisItem, *args, **kwargs):
            self._axis: AxisItem = axis
            super(Axis._Property, self).__init__(*args, **kwargs)

    class _Label(_Property, PropertyBase[str]):

        def _get(self, **kwargs) -> str:
            return self._axis.label

        def _set(self, value: str, **kwargs) -> None:
            self._axis.setLabel(value, **kwargs)

    class _Range(_Property, PropertyBase[Tuple[Optional[float], Optional[float]]]):

        def __init__(self, axes: PlotItem, axis: AxisItem, *args, **kwargs):
            self._axes: PlotItem = axes
            super(Axis._Range, self).__init__(axis, *args, **kwargs)

        def _get(self, **kwargs) -> Tuple[float, float]:
            return tuple(self._axis.range)

        def _set(self, value: Tuple[Optional[float], Optional[float]], **kwargs) -> None:
            raise NotImplementedError

        def _get_fact_value(self, value: Tuple[Optional[float], Optional[float]]) -> Tuple[float, float]:
            value = (self._axis.range[0] if value[0] is None else value[0],
                     self._axis.range[1] if value[1] is None else value[1])
            value = (value[0] or self._axis.range[0], value[1] or self._axis.range[1])
            if value[0] == -np.inf or value[1] == np.inf:
                self._axes.autoRange()
            return (self._axis.range[0] if value[0] == -np.inf else value[0],
                    self._axis.range[1] if value[1] == np.inf else value[1])

    class _Scale(_Property, PropertyBase[Scale]):

        def __init__(self, axes: PlotItem, axis: AxisItem, *args, **kwargs):
            self._axes: PlotItem = axes
            super(Axis._Scale, self).__init__(axis, *args, **kwargs)

        def _get(self, **kwargs) -> Scale:
            if self._axis.logMode:
                return Scale.Log
            else:
                return Scale.Linear

        def _set(self, value: Scale, **kwargs) -> None:
            if value == Scale.Linear:
                self._set_scale(False)
            elif value == Scale.Log:
                self._set_scale(True)
            else:
                raise NotImplementedError

        def _set_scale(self, log_mode: bool) -> None:
            raise NotImplementedError

    class _ScaleLine(_Property, PropertyBase[LineStyle]):

        def _get(self, **kwargs) -> LineStyle:
            value = self._axis.grid
            if value == 255:
                return LineStyle.Solid
            elif not value:
                return LineStyle.Nothing
            else:
                raise NotImplementedError

        def _set(self, value: LineStyle, **kwargs) -> None:
            if value == LineStyle.Solid:
                self._axis.setGrid(255)
            elif value == LineStyle.Nothing:
                self._axis.setGrid(False)
            else:
                raise NotImplementedError

    # endregion

    def __init__(self, axes: PlotItem, direction: Direction,
                 range_type: Type[_Range],
                 scale_type: Type[_Scale], *args, **kwargs):

        self._axes: PlotItem = axes
        self._raw: AxisItem = axes.getAxis(self._axis_name())

        self._label: Axis._Label = Axis._Label(self._raw)
        self._scale: Axis._Scale = scale_type(self._axes, self._raw)
        self._range: Axis._Range = range_type(self._axes, self._raw)

        self._scale_line: Axis._ScaleLine = Axis._ScaleLine(self._raw)

        # self._ticks: Axis._Ticks = Axis._Ticks(self._raw, alt=(np.array([]), np.array([])))
        self._ticks: np.ndarray = np.array([])
        self._tick_labels: np.ndarray = np.array([])
        self._minor_ticks: np.ndarray = np.array([])
        self._minor_tick_labels: np.ndarray = np.array([])

        self._tick_raw: List[Tuple[float, Any]] = []

        super(Axis, self).__init__(direction, *args, **kwargs)

    # region Private Method

    def _axis_name(self):
        raise NotImplementedError

    def _set_tick_raw(self, *,
                      ticks: Optional[np.ndarray] = None, tick_labels: Optional[np.ndarray] = None,
                      minor_ticks: Optional[np.ndarray] = None, minor_tick_labels: Optional[np.ndarray] = None):
        if ticks is not None:
            self._ticks = ticks
        if tick_labels is not None:
            self._tick_labels = tick_labels
        if minor_ticks is not None:
            self._minor_ticks = minor_ticks
        if minor_tick_labels is not None:
            self._minor_tick_labels = minor_tick_labels

        self._tick_raw = [
            [v for v in zip(self._ticks, self._tick_labels)],
            [v for v in zip(self._minor_ticks, self._minor_tick_labels)]
        ]
        self._raw.setTicks(self._tick_raw)

    # endregion

    # region AxisBase Method

    def label(self, value: Optional[str] = None, **kwargs) -> str:
        return self._label(value, **kwargs)

    def scale(self, value: Optional[Scale] = None, **kwargs) -> Scale:
        return self._scale(value)

    def range(self, value: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[float, float]:
        return self._range(value)

    def scale_line(self, value: Optional[LineStyle] = None, **kwargs) -> LineStyle:
        return self._scale_line(value, **kwargs)

    def minor_scale_line(self, value: Optional[LineStyle] = None, **kwargs) -> LineStyle:
        return self._scale_line()

    def ticks(self, value: Optional[np.ndarray] = None, **kwargs) -> Optional[np.ndarray]:
        if value is not None:
            self._set_tick_raw(ticks=value, tick_labels=value)
        return self._ticks

    def minor_ticks(self, value: Optional[np.ndarray] = None, **kwargs) -> Optional[np.ndarray]:
        if value is not None:
            self._set_tick_raw(minor_ticks=value, minor_tick_labels=value)
        return self._minor_ticks

    def tick_labels(self, value: Optional[np.ndarray] = None, **kwargs) -> Optional[np.ndarray]:
        if value is not None:
            self._set_tick_raw(tick_labels=value)
        return self._tick_labels

    def minor_tick_labels(self, value: Optional[np.ndarray] = None, **kwargs) -> Optional[np.ndarray]:
        if value is not None:
            self._set_tick_raw(minor_ticks=value, minor_tick_labels=value)
        return self._minor_tick_labels

    # endregion

    # region GraphBase Method

    def clear(self) -> None:
        self._axes.hideAxis(self._axis_name())

    # endregion
