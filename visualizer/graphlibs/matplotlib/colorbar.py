from abc import ABC
from typing import Optional, Tuple

import numpy as np

from matplotlib.colorbar import Colorbar as ColorBarRaw
from matplotlib.figure import Figure as FigureRaw
# from matplotlib.cm import ScalarMappable

from .. import ColorBarBase, PropertyBase


class ColorBar(ColorBarBase):

    class Property(PropertyBase, ABC):

        def __init__(self, color_bar: ColorBarRaw, *args, **kwargs):
            self._color_bar: ColorBarRaw = color_bar
            super(ColorBar.Property, self).__init__(*args, **kwargs)

    class _Range(Property, PropertyBase[Tuple[Optional[float], Optional[float]]]):

        def _get(self, **kwargs) -> Tuple[Optional[float], Optional[float]]:
            return self._color_bar.mappable.get_clim()

        def _set(self, value: Tuple[Optional[float], Optional[float]], **kwargs) -> None:
            if value[0] == -np.inf or value[1] == np.inf:
                a = self._color_bar.mappable.get_array()
                value = (np.min(a) if value[0] == -np.inf else value[0],
                         np.max(a) if value[1] == np.inf else value[1])
            self._color_bar.mappable.set_clim(value[0], value[1], **kwargs)

    def __init__(self, color_bar: ColorBarRaw, figure: FigureRaw, *args, **kwargs):

        self._raw: ColorBarRaw = color_bar
        self._figure: FigureRaw = figure

        self._range: ColorBar._Range = ColorBar._Range(self._raw)

        super(ColorBar, self).__init__(*args, **kwargs)

    def range(self, value: Optional[Tuple[Optional[float], Optional[float]]] = None, **kwargs) -> Tuple[float, float]:
        return self._range(value, **kwargs)

    def clear(self) -> None:
        self._figure.remove()
