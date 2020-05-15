from typing import Optional, Tuple

import numpy as np

from matplotlib.axes import Axes as AxesRaw
from matplotlib.axis import Axis as AxisRaw

from .. import AxisBase, LineStyle, PropertyBase

from .axis import Axis
from .style import line_style_to_str


class AxisY(Axis):

    # region Class

    class _Label(Axis.Property, PropertyBase[str]):

        # def __call__(self, value: Optional[str] = None, **kwargs) -> str:
        #     return super(AxisY._Label, self).__call__(value, **kwargs)

        def _get(self, **kwargs) -> str:
            return self._axes.get_ylabel()

        def _set(self, value: str, **kwargs) -> None:
            self._axes.set_ylabel(value, **kwargs)

    class _ScaleLine(Axis.Property, PropertyBase[LineStyle]):

        def _get(self, **kwargs) -> None:
            return None

        def _set(self, value: LineStyle, **kwargs) -> None:
            if value == LineStyle.Nothing:
                self._axes.grid(b=False, which="major", axis="y")
            else:
                self._axes.grid(which="major", axis="y", linestyle=line_style_to_str(value), **kwargs)

    class _Ticks(Axis.Property, PropertyBase[np.ndarray]):

        def _get(self, **kwargs) -> np.ndarray:
            return np.array(self._axes.get_yticks())

        def _set(self, value: np.ndarray, **kwargs) -> None:
            self._axes.set_yticks(value)

    class _TickLabels(Axis.Property, PropertyBase[np.ndarray]):

        def _get(self, **kwargs) -> np.ndarray:
            return np.array(self._axes.get_yticklabels())

        def _set(self, value: np.ndarray, **kwargs) -> None:
            self._axes.set_yticklabels(value)

    class _MinorScaleLine(Axis.Property, PropertyBase[LineStyle]):

        def _get(self, **kwargs) -> None:
            return None

        def _set(self, value: LineStyle, **kwargs) -> None:
            if value == LineStyle.Nothing:
                self._axes.grid(b=False, which="minor", axis="y")
            else:
                self._axes.grid(which="minor", axis="y", linestyle=line_style_to_str(value), **kwargs)

    class _MinorTicks(Axis.Property, PropertyBase[np.ndarray]):

        def _get(self, **kwargs) -> np.ndarray:
            return np.array(self._axes.get_yticks(minor=True))

        def _set(self, value: np.ndarray, **kwargs) -> None:
            self._axes.set_yticks(value, minor=True)

    class _MinorTickLabels(Axis.Property, PropertyBase[np.ndarray]):

        def _get(self, **kwargs) -> np.ndarray:
            return np.array(self._axes.get_yticklabels(minor=True))

        def _set(self, value: np.ndarray, **kwargs) -> None:
            self._axes.set_yticklabels(value, minor=True)

    # endregion

    def __init__(self, axis: AxisRaw, axes: AxesRaw, *args, **kwargs):
        super(AxisY, self).__init__(axis, axes, AxisBase.Direction.Y, AxisY._Label,
                                    AxisY._ScaleLine, AxisY._Ticks, AxisY._TickLabels,
                                    AxisY._MinorScaleLine, AxisY._MinorTicks, AxisY._MinorTickLabels,
                                    *args, **kwargs)

    def clear(self):
        self._ax.set_ylim(auto=True)
        self._clear()

    # region AxisBase Method

    def range(self, value: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[float, float]:
        if value is not None:
            if value[0] == -np.inf or value[1] == np.inf:
                self._ax.set_ylim(auto=True)
            self._ax.set_ylim(self._ax.get_ylim()[0] if value[0] == -np.inf else value[0],
                              self._ax.get_ylim()[1] if value[1] == np.inf else value[1])
        return self._ax.get_ylim()

    # endregion

    # region Private Method

    def _get_scale_str(self) -> str:
        return self._ax.get_yscale()

    def _set_scale_str(self, value: str) -> None:
        self._ax.set_yscale(value)

    def _set_scale_func(self, scale_func: Axis.ScaleFunctionABC) -> None:
        self._ax.set_yscale("function", functions=(scale_func.function_tuple()))

    # endregion
