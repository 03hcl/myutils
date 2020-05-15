from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np

from matplotlib.axes import Axes as AxesRaw
from matplotlib.axis import Axis as AxisRaw
import matplotlib.ticker as ticker

from .. import AxisBase, LineStyle, PropertyBase

Scale = AxisBase.Scale
Direction = AxisBase.Direction


class Axis(AxisBase):

    # region Class, Dict

    class ScaleFunctionABC(ABC):

        @abstractmethod
        def forward(self, x: float) -> float:
            pass

        @abstractmethod
        def inverse(self, x: float) -> float:
            pass

        def function_tuple(self) -> Tuple[Callable[[float], float], Callable[[float], float]]:
            return lambda x: self.forward(x), lambda x: self.inverse(x)

    class Power(ScaleFunctionABC):

        def __init__(self, a: float):
            self._a: float = a

        def forward(self, x: float) -> float:
            return x ** self._a

        def inverse(self, x: float) -> float:
            return x ** (1 / self._a)

    class Property:
        # noinspection PyUnusedLocal
        def __init__(self, axes: AxesRaw, *args, **kwargs):
            self._axes: AxesRaw = axes

    ScaleStrDict: Dict[Scale, str] = {
        Scale.Linear: "linear",
        Scale.Log: "log",
        Scale.SymmetricLog: "symlog",
        Scale.Logit: "logit",
    }

    # endregion

    def __init__(self, axis: AxisRaw, axes: AxesRaw, direction: Direction,
                 label_type: Type[PropertyBase[str]],
                 scale_line_type: Type[PropertyBase[LineStyle]], ticks_type: Type[PropertyBase[np.ndarray]],
                 tick_labels_type: Type[PropertyBase[np.ndarray]],
                 minor_scale_line_type: Type[PropertyBase[LineStyle]], minor_ticks_type: Type[PropertyBase[np.ndarray]],
                 minor_tick_labels_type: Type[PropertyBase[np.ndarray]],
                 *args, **kwargs):

        self._raw: AxisRaw = axis
        self._ax: AxesRaw = axes

        self._label: PropertyBase[str] = label_type(axes)

        self._scale: Scale = Scale.Unknown
        scale_str: str = self._get_scale_str()
        for k, v in Axis.ScaleStrDict.items():
            if v == scale_str:
                self._scale = k

        self._scale_line: PropertyBase[LineStyle] = scale_line_type(axes, alt=LineStyle.Unknown)
        self._ticks: PropertyBase[np.ndarray] = ticks_type(axes)
        self._tick_labels: PropertyBase[np.ndarray] = tick_labels_type(axes)

        self._minor_scale_line: PropertyBase[LineStyle] = minor_scale_line_type(axes, alt=LineStyle.Unknown)
        self._minor_ticks: PropertyBase[np.ndarray] = minor_ticks_type(axes)
        self._minor_tick_labels: PropertyBase[np.ndarray] = minor_tick_labels_type(axes)

        super(Axis, self).__init__(direction, *args, **kwargs)

    # region AxisBase Method

    def label(self, value: Optional[str] = None, **kwargs) -> str:
        return self._label(value, **kwargs)

    def scale(self, value: Optional[Scale] = None, **kwargs) -> Scale:
        if value is not None:
            if value in Axis.ScaleStrDict:
                self._set_scale_str(Axis.ScaleStrDict[value])
            elif value == Scale.Power:
                self._set_scale_func(Axis.Power(kwargs["a"]))
            else:
                raise NotImplementedError
            self._ticks.update()
            self._tick_labels.update()
            self._minor_ticks.update()
            self._minor_tick_labels.update()
            self._scale = value
        return self._scale

    def scale_line(self, value: Optional[LineStyle] = None, **kwargs) -> LineStyle:
        return self._scale_line(value, **kwargs)

    def minor_scale_line(self, value: Optional[LineStyle] = None, **kwargs) -> LineStyle:
        return self._minor_scale_line(value, **kwargs)

    def ticks(self, value: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self._ticks(value, **kwargs)

    def tick_labels(self, value: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self._tick_labels(value, **kwargs)

    def minor_ticks(self, value: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self._minor_ticks(value, **kwargs)

    def minor_tick_labels(self, value: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self._minor_tick_labels(value, **kwargs)

    # endregion

    # region Private Method

    def _clear(self):

        self._raw.set_major_locator(ticker.AutoLocator())
        self._raw.set_minor_locator(ticker.NullLocator())

        if self._scale == Scale.Log:
            self._raw.set_major_locator(ticker.LogLocator())
            self._raw.set_minor_locator(ticker.LogLocator(subs="auto"))
        elif self._scale == Scale.Logit:
            self._raw.set_major_locator(ticker.LogitLocator())
        elif self._scale == Scale.SymmetricLog:
            self._raw.set_major_locator(ticker.SymmetricalLogLocator())

    # endregion

    # region Abstract Method

    @abstractmethod
    def _get_scale_str(self) -> str:
        pass

    @abstractmethod
    def _set_scale_str(self, value: str) -> None:
        pass

    @abstractmethod
    def _set_scale_func(self, scale_func: ScaleFunctionABC) -> None:
        pass

    # endregion
