from typing import Optional, Tuple

import numpy as np

from matplotlib.axes import Axes as AxesRaw
from matplotlib.axis import Axis as AxisRaw

from .. import AxisBase, LineStyle, PropertyBase

from .axis import Axis
from .style import line_style_to_str


class AxisX(Axis):

    # region Class

    class _Label(Axis.Property, PropertyBase[str]):

        # def __call__(self, value: Optional[str] = None, **kwargs) -> str:
        #     return super(AxisX._Label, self).__call__(value, **kwargs)

        def _get(self, **kwargs) -> str:
            return self._axes.get_xlabel()

        def _set(self, value: str, **kwargs) -> None:
            self._axes.set_xlabel(value, **kwargs)

    class _ScaleLine(Axis.Property, PropertyBase[LineStyle]):

        def _get(self, **kwargs) -> None:
            return None

        def _set(self, value: LineStyle, **kwargs) -> None:
            if value == LineStyle.Nothing:
                self._axes.grid(b=False, which="major", axis="x")
            else:
                self._axes.grid(which="major", axis="x", linestyle=line_style_to_str(value), **kwargs)

    class _Ticks(Axis.Property, PropertyBase[np.ndarray]):

        def _get(self, **kwargs) -> np.ndarray:
            return np.array(self._axes.get_xticks())

        def _set(self, value: np.ndarray, **kwargs) -> None:
            self._axes.set_xticks(value)

    class _TickLabels(Axis.Property, PropertyBase[np.ndarray]):

        def _get(self, **kwargs) -> np.ndarray:
            return np.array(self._axes.get_xticklabels())

        def _set(self, value: np.ndarray, **kwargs) -> None:
            self._axes.set_xticklabels(value)

    class _MinorScaleLine(Axis.Property, PropertyBase[LineStyle]):

        def _get(self, **kwargs) -> None:
            return None

        def _set(self, value: LineStyle, **kwargs) -> None:
            if value == LineStyle.Nothing:
                self._axes.grid(b=False, which="minor", axis="x")
            else:
                self._axes.grid(which="minor", axis="x", linestyle=line_style_to_str(value), **kwargs)

    class _MinorTicks(Axis.Property, PropertyBase[np.ndarray]):

        def _get(self, **kwargs) -> np.ndarray:
            return np.array(self._axes.get_xticks(minor=True))

        def _set(self, value: np.ndarray, **kwargs) -> None:
            self._axes.set_xticks(value, minor=True)

    class _MinorTickLabels(Axis.Property, PropertyBase[np.ndarray]):

        def _get(self, **kwargs) -> np.ndarray:
            return np.array(self._axes.get_xticklabels(minor=True))

        def _set(self, value: np.ndarray, **kwargs) -> None:
            self._axes.set_xticklabels(value, minor=True)

    # endregion

    def __init__(self, axis: AxisRaw, axes: AxesRaw, **kwargs):
        super(AxisX, self).__init__(axis, axes, AxisBase.Direction.X, AxisX._Label,
                                    AxisX._ScaleLine, AxisX._Ticks, AxisX._TickLabels,
                                    AxisX._MinorScaleLine, AxisX._MinorTicks, AxisX._MinorTickLabels,
                                    **kwargs)

    def clear(self):
        self._ax.set_xlim(auto=True)
        self._clear()

    # region AxisBase Method

    def range(self, value: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[float, float]:
        if value is not None:
            if value[0] == -np.inf or value[1] == np.inf:
                self._ax.set_xlim(auto=True)
            self._ax.set_xlim(self._ax.get_xlim()[0] if value[0] == -np.inf else value[0],
                              self._ax.get_xlim()[1] if value[1] == np.inf else value[1])
        return self._ax.get_xlim()

    # endregion

    # region Private Method

    def _get_scale_str(self) -> str:
        return self._ax.get_xscale()

    def _set_scale_str(self, value: str) -> None:
        self._ax.set_xscale(value)

    def _set_scale_func(self, scale_func: Axis.ScaleFunctionABC) -> None:
        self._ax.set_xscale("function", functions=(scale_func.function_tuple()))

    # endregion
