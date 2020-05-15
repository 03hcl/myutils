from typing import Optional, Tuple

from pyqtgraph import PlotItem

from .axis import Axis


class AxisY(Axis):

    class _Scale(Axis._Scale):

        def _set_scale(self, log_mode: bool) -> None:
            self._axes.setLogMode(y=log_mode)
            self._axis.setLogMode(log_mode)

    class _Range(Axis._Range):

        def _set(self, value: Tuple[Optional[float], Optional[float]], **kwargs) -> None:
            value = self._get_fact_value(value)
            self._axes.setYRange(value[0], value[1], padding=0)
            # self._axes.setRange(yRange=(value[0] or self._axis.range[0], value[1] or self._axis.range[1]),
            #                     padding=0)

    def __init__(self, axes: PlotItem, *args, **kwargs):
        super(AxisY, self).__init__(axes, Axis.Direction.Y,
                                    AxisY._Range, AxisY._Scale,
                                    *args, **kwargs)

    def _axis_name(self):
        return "left"
